"""Run ber5 on one EC2 machine from your laptop (needs: pip install boto3 paramiko; AWS keys in ~/.aws).

python aws_ber5.py quota [request]                     on-demand standard vCPU quota (request 96)
python aws_ber5.py launch TYPE [DISK_GB] [MAX_MIN]     Ubuntu 24.04; powers off = terminates after MAX_MIN
python aws_ber5.py setup --kaggle-dataset OWNER/SLUG   install + download the dataset with your ~/.kaggle token
python aws_ber5.py setup --local-data DIR              ... or upload a local student_resource/dataset folder
python aws_ber5.py run TAG 'CFG_JSON' HOURS            start ber5 in the background (log: ~/run_TAG.log)
python aws_ber5.py log TAG [LINES]                     tail the log
python aws_ber5.py fetch TAG LOCAL_DIR                 download outputs + files needed for ensembling
python aws_ber5.py extend MINUTES                      reset the auto-terminate timer
python aws_ber5.py sh "CMD" | status | terminate

Keys and state live in ~/.ber5_aws (never in the repo). Machines stop billing when terminated.
"""
import json
import os
import sys
import time
import urllib.request

import boto3

HOME = os.path.join(os.path.expanduser("~"), ".ber5_aws")
STATE = os.path.join(HOME, "state.json")
KEY = os.path.join(HOME, "ber-key.pem")
REGION = os.environ.get("AWS_REGION", "us-east-1")
BER5 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "ber5.py")
DATA = "/home/ubuntu/data/student_resource/dataset"
PRICE = {"r7i.4xlarge": 1.06, "r7i.8xlarge": 2.12, "r7i.16xlarge": 4.23, "r7i.24xlarge": 6.35,
         "c7i.16xlarge": 2.86, "c7i.24xlarge": 4.28, "m7i.16xlarge": 3.23, "m7i.24xlarge": 4.84}


def state():
    return json.load(open(STATE)) if os.path.exists(STATE) else {}


def save(st):
    os.makedirs(HOME, exist_ok=True)
    json.dump(st, open(STATE, "w"), indent=1)


def ec2():
    return boto3.client("ec2", region_name=REGION)


def connect():
    import paramiko
    st = state()
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    last = None
    for _ in range(30):
        try:
            c.connect(st["ip"], username="ubuntu", key_filename=KEY, timeout=20, banner_timeout=30)
            return c
        except Exception as e:  # still booting
            last = e
            time.sleep(10)
    raise SystemExit(f"ssh failed: {last}")


def sh(cmd):
    c = connect()
    _, out, _ = c.exec_command(cmd, get_pty=True)
    for line in iter(out.readline, ""):
        print(line, end="", flush=True)
    code = out.channel.recv_exit_status()
    c.close()
    return code


def put_dir(sftp, local, remote):
    for root, _, files in os.walk(local):
        rel = os.path.relpath(root, local).replace("\\", "/")
        target = remote if rel == "." else f"{remote}/{rel}"
        sh(f"mkdir -p {target}")
        for f in files:
            print("upload", os.path.join(root, f), flush=True)
            sftp.put(os.path.join(root, f), f"{target}/{f}")


def main():
    cmd, args = sys.argv[1], sys.argv[2:]
    st = state()
    if cmd == "quota":
        sq = boto3.client("service-quotas", region_name=REGION)
        q = sq.get_service_quota(ServiceCode="ec2", QuotaCode="L-1216C47A")["Quota"]["Value"]
        print("on-demand standard vCPU quota:", q)
        if q < 96 and args[:1] == ["request"]:
            r = sq.request_service_quota_increase(ServiceCode="ec2", QuotaCode="L-1216C47A", DesiredValue=96.0)
            print("requested 96:", r["RequestedQuota"]["Status"])
    elif cmd == "launch":
        itype, disk = args[0], int(args[1]) if len(args) > 1 else 200
        max_min = int(args[2]) if len(args) > 2 else 600
        e = ec2()
        os.makedirs(HOME, exist_ok=True)
        if not os.path.exists(KEY):
            try:
                e.delete_key_pair(KeyName="ber-key")
            except Exception:
                pass
            open(KEY, "w").write(e.create_key_pair(KeyName="ber-key", KeyType="ed25519")["KeyMaterial"])
        if "sg" not in st:
            vpc = e.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"][0]["VpcId"]
            try:
                st["sg"] = e.create_security_group(GroupName="ber-ssh", Description="ssh for ber5", VpcId=vpc)["GroupId"]
            except Exception:
                st["sg"] = e.describe_security_groups(GroupNames=["ber-ssh"])["SecurityGroups"][0]["GroupId"]
        ip = urllib.request.urlopen("https://checkip.amazonaws.com").read().decode().strip()
        try:
            e.authorize_security_group_ingress(GroupId=st["sg"], IpPermissions=[{
                "IpProtocol": "tcp", "FromPort": 22, "ToPort": 22, "IpRanges": [{"CidrIp": f"{ip}/32"}]}])
        except Exception:
            pass  # rule already there
        imgs = e.describe_images(Owners=["099720109477"], Filters=[
            {"Name": "name", "Values": ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]},
            {"Name": "state", "Values": ["available"]}])["Images"]
        ami = max(imgs, key=lambda i: i["CreationDate"])["ImageId"]
        r = e.run_instances(ImageId=ami, InstanceType=itype, MinCount=1, MaxCount=1, KeyName="ber-key",
                            SecurityGroupIds=[st["sg"]], UserData=f"#!/bin/bash\nshutdown -h +{max_min}\n",
                            BlockDeviceMappings=[{"DeviceName": "/dev/sda1", "Ebs": {
                                "VolumeSize": disk, "VolumeType": "gp3", "DeleteOnTermination": True}}],
                            InstanceInitiatedShutdownBehavior="terminate",
                            TagSpecifications=[{"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": "ber5"}]}])
        iid = r["Instances"][0]["InstanceId"]
        e.get_waiter("instance_running").wait(InstanceIds=[iid])
        d = e.describe_instances(InstanceIds=[iid])["Reservations"][0]["Instances"][0]
        st.update(instance=iid, ip=d["PublicIpAddress"], type=itype, launched=time.time())
        save(st)
        print(f"running {iid} {itype} {d['PublicIpAddress']}; auto-terminates in {max_min} min")
    elif cmd == "setup":
        c = connect()
        sftp = c.open_sftp()
        sftp.put(BER5, "/home/ubuntu/ber5.py")
        base = ("set -e; nproc; free -g | head -2; sudo apt-get -qq update >/dev/null; "
                "sudo DEBIAN_FRONTEND=noninteractive apt-get -qq install -y python3-venv unzip >/dev/null; "
                "python3 -m venv ~/venv; ~/venv/bin/pip -q install numpy==2.0.2 pandas==2.3.3 numba==0.60.0 "
                "lightgbm==4.6.0 rapidfuzz==3.14.6 anyascii==0.3.3 psutil kaggle")
        if "--kaggle-dataset" in args:
            slug = args[args.index("--kaggle-dataset") + 1]
            sh("mkdir -p ~/.kaggle ~/data")
            sftp.put(os.path.expanduser("~/.kaggle/kaggle.json"), "/home/ubuntu/.kaggle/kaggle.json")
            code = sh(base + f"; chmod 600 ~/.kaggle/kaggle.json; cd ~/data && ~/venv/bin/kaggle datasets download "
                             f"-q {slug} && unzip -q -o *.zip && rm -f *.zip; find ~/data -name '*.tsv' | head")
        else:
            put_dir(sftp, args[args.index("--local-data") + 1], DATA)
            code = sh(base + f"; find {DATA} -name '*.tsv' | head")
        sftp.close()
        c.close()
        sys.exit(code)
    elif cmd == "run":
        tag, cfg, hours = args[0], args[1], args[2]
        json.loads(cfg)
        c = connect()
        c.open_sftp().put(BER5, f"/home/ubuntu/ber5_{tag}.py")
        c.close()
        sys.exit(sh(f"cd ~ && D=$(dirname $(dirname $(find ~/data -name train_source1.tsv | head -1))) && "
                    f"nohup ~/venv/bin/python -u ber5_{tag}.py --data $D --work ~/work_{tag} --stage all "
                    f"--cfg '{cfg}' --budget-hours {hours} > ~/run_{tag}.log 2>&1 & sleep 5; tail -3 ~/run_{tag}.log"))
    elif cmd == "log":
        sys.exit(sh(f"tail -n {args[1] if len(args) > 1 else 40} ~/run_{args[0]}.log"))
    elif cmd == "fetch":
        tag, local = args[0], args[1]
        os.makedirs(os.path.join(local, "model"), exist_ok=True)
        os.makedirs(os.path.join(local, "output"), exist_ok=True)
        c = connect()
        sftp = c.open_sftp()
        w = f"/home/ubuntu/work_{tag}"
        names = [f"output/{f}" for f in sftp.listdir(f"{w}/output")] + \
                [f"model/{f}" for f in sftp.listdir(f"{w}/model")] + \
                [f for f in sftp.listdir(w) if f.endswith(".npy")]
        for f in names:
            print("get", f, flush=True)
            sftp.get(f"{w}/{f}", os.path.join(local, f))
        sftp.get(f"/home/ubuntu/run_{tag}.log", os.path.join(local, f"run_{tag}.log"))
        c.close()
    elif cmd == "extend":
        sys.exit(sh(f"sudo shutdown -c; sudo shutdown -h +{int(args[0])}"))
    elif cmd == "sh":
        sys.exit(sh(args[0]))
    elif cmd == "status":
        if "instance" not in st:
            print("no instance")
            return
        d = ec2().describe_instances(InstanceIds=[st["instance"]])["Reservations"][0]["Instances"][0]
        hrs = (time.time() - st["launched"]) / 3600
        print(st["instance"], d["State"]["Name"], st["type"], st["ip"], f"{hrs:.2f} h",
              f"~${hrs * PRICE.get(st['type'], 0):.2f}")
    elif cmd == "terminate":
        if "instance" in st:
            ec2().terminate_instances(InstanceIds=[st["instance"]])
            print("terminated", st.pop("instance"))
            save(st)


if __name__ == "__main__":
    main()

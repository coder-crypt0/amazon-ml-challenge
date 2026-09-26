# Running ber5 on AWS (big machine)

`src/ber5.py` is the whole pipeline (train + test + validation report). On Kaggle (4 CPU, 31 GB) one run takes
~7 h. On an EC2 `r7i.16xlarge` (64 vCPU, 512 GB, ~$4.2/h) it takes ~2–3 h and can use far more data.

## 0. One-time setup on your laptop (Python 3.10+)
```sh
git fetch && git checkout feat/ber5-v5
pip install boto3 paramiko
```
- AWS access key in `~/.aws/credentials` (IAM user with AmazonEC2FullAccess + ServiceQuotasFullAccess):
  ```
  [default]
  aws_access_key_id = AKIA...
  aws_secret_access_key = ...
  ```
- **Data**, one of:
  - **(a)** Kaggle dataset `aditya32211/student-resource`. Aditya shares it with your Kaggle account, and you
    put your token in `~/.kaggle/kaggle.json`.
  - **(b)** Your own copy of `student_resource/dataset` (contains `train/` and `test/`).

All commands below run from `code/business_entity_resolution/scripts/`.

## 1. CPU limit (new accounts are capped)
```sh
python aws_ber5.py quota request      # asks for 96 vCPUs; approval takes minutes to hours
```
Until approved, use the largest size the quota allows, e.g. `r7i.8xlarge` (32 vCPU, 256 GB).

## 2. Launch + install (~5 min)
```sh
python aws_ber5.py launch r7i.16xlarge 200 600    # 200 GB disk; the machine TERMINATES ITSELF after 600 min
python aws_ber5.py setup --kaggle-dataset aditya32211/student-resource
#   or: python aws_ber5.py setup --local-data "C:/path/to/student_resource/dataset"
```

## 3. Run (in the background on the machine)
```sh
python aws_ber5.py run big '{"norm":2,"sample":0.6,"stage2":true,"tcomp":true,"cprior":true,"neg_keep":0.3,"budget_s1":40000,"budget_t":120000,"k_rev":10,"k_fwd":16,"k_name":8,"k_addr":8,"chunk":6000000}' 9
python aws_ber5.py log big            # progress; look for "STAGE 1 REPORT" / "STAGE 2 REPORT" (val_f05)
```
- A second, *diverse* run for the ensemble can follow on the same machine. Start it after `big` finishes, or
  in parallel on a 96-core machine:
  ```sh
  python aws_ber5.py run div '{"norm":2,"sample":0.6,"stage2":true,"tcomp":true,"cprior":true,"neg_keep":0.3,"budget_s1":40000,"budget_t":120000,"k_rev":10,"k_fwd":16,"k_name":8,"k_addr":8,"chunk":6000000,"leaves":511,"min_leaf":200,"ff":0.6,"bf":0.7,"lr":0.05,"model_seed":7}' 9
  ```
- Need more time than the auto-terminate timer? Run `python aws_ber5.py extend 300`.

## 4. Fetch results, ensemble, terminate
```sh
python aws_ber5.py fetch big runs/big
python aws_ber5.py fetch div runs/div
python aws_ber5.py terminate          # stops billing; ALWAYS do this
python ens.py --data <dataset dir> --out runs/ens --runs runs/big runs/div
```
- Upload `runs/ens/matching_results.tsv` (or `runs/big/output/matching_results.tsv`) to the portal.
- `model/meta.json` of each run holds the validation macro F0.5 (`report`, `report2`) and the threshold.

## Safety / cost
- Every machine powers off and terminates after the `launch` timer (default 600 min), even if your laptop
  disconnects. `python aws_ber5.py status` prints the running time and approximate cost.
- The pipeline's own `--budget-hours` (last argument of `run`) stops scoring early and still writes valid files.
- Keys and state stay in `~/.ber5_aws` on your laptop, never in the repo.

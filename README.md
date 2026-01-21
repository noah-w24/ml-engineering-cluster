# ml-engineering-cluster

```sh
kubectl create secret generic sops-age -n argocd --from-file=age.agekey=age.agekey
```

```sh
find secrets -name "*.sops.yaml" -exec sops -e -i {} \;
```

```sh
export SOPS_AGE_KEY_FILE="/Users/i588313/Documents/GitHub/ml-engineering-cluster/age.agekey"
```

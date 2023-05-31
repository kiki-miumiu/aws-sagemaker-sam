# sam-boilerplate

This is a boilerplate repo for aws sam

## Prerequisites
* A VPC with Subnets

## Deploy Local

```bash
aws sts get-caller-identity
sam build && sam deploy --guided
```

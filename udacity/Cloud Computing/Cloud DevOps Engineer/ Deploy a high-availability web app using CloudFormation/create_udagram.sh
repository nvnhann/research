 aws cloudformation create-stack --stack-name udagram-iac-server \
--template-body file://templates/udagram.yml   \
--parameters file://templates/udagram-parameters.json  \
--capabilities "CAPABILITY_NAMED_IAM"  \
--region=us-west-2
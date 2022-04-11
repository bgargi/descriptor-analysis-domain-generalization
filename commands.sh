
# TRAIN ON ERM + DOMAINNET

# python3 -m domainbed.scripts.train\
#        --data_dir=./domainbed/data/\
#        --output_dir=./domainbed/erm/\
#        --algorithm ERM\
#        --dataset DomainNet 


# TRAIN ON MIXUP + DOMAINNET

# python3 -m domainbed.scripts.train\
#        --data_dir=./domainbed/data/\
#        --output_dir=./domainbed/mixup/\
#        --algorithm Mixup\
#        --dataset DomainNet 

# Extract Descriptor for certain ERM
# python3 descriptor_extraction.py --data_dir=./domainbed/data/ --output_dir=./domainbed/ermnew/ --algorithm ERM --dataset DomainNet

# DOWNLOAD DomainNet, for others, comment out that line in datasets.py
# python3 -m domainbed.scripts.download \
#        --data_dir=./domainbed/data

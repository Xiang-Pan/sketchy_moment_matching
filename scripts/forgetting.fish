python -u main.py 
--fraction 0.1 \
--dataset CIFAR10\
--data_path ~/datasets \
--num_exp 5 --workers 10 \
--optimizer SGD \
-se 10 --selection Glister \
--model InceptionV3 --lr 0.1 -sp ./result --batch 128

python main.py -m \
selection.method=Glister \
selection.frac=0.1 \
selection.selection_epochs=10 \
selection.model=resnet50-swav_run0 \



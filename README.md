###
1. The project is inspired from [A Closer Look at Few-Shot Crosslingual Transfer: The Choice of Shots Matters](https://arxiv.org/pdf/2012.15682.pdf), the original code could be found in the [repo](https://github.com/fsxlt/code)

### Data 
1. from xtreme
2. data is ignored due to the size
3. the prediction of test data needs to be set to X everytime before running
    - all data from xtreme
4. Due to the size of the dataset, the language datasets are not included in the github repo.
5. <em>sent_pieces</em> is the segmented sentences

### The running process
1. Running platform: google colab
2. finetuning $\rightarrow$ adapt_learning

### Active learning
1. Use the library, modAL
2. pool-based sampling + uncertainty sampling
3. The basic idea is replacing the random sampling process in wrap_sampler.py in the original project with active learning. Hopefully this could increase the accuracy and improve the performance.
4. we use the <em>input_idses</em> and <em>tags_ides</em> from language_dataset for active learning (tensor format)

### Notes
1. get the tags from the previous samples
2. use the previous project to get the probability
3. set the samples according to the previous project
4. train 
5. the training, develop and test set are for evaluation, instead of training.

   
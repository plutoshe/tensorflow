
## RNN in Android

It contains a simple rnn demo from training to android application utilizing tensorflow.

### Python Client

1. model training

Run command ```python3 ptb_word_lm.py --data_path=<raw_text_data_dir_path> --mode=train``` under ```androidnn/model_train/```. The simple data required is in the data/ directory of the PTB dataset from Tomas Mikolov's webpage: ```http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz```.

It generates model ```graph_train*.pb``` and dictionary file ```dict.txt```.

2. model testing

Run command ```python3 ptb_word_lm.py --data_path=<raw_text_data_dir_path> --mode=test --graph_path=<model_graph>```

3. sequence prediction

Run command ```python3 ptb_word_lm.py --data_path=<raw_text_data_dir_path> --mode=predict --graph_path=<model_graph>```. Input history sequnce, it will predict the whole sentence.

### Android Application

The application leverage on bazel and tensorflow. As a prerequisite, Bazel, the Android NDK, and the Android SDK must all be installed on your system. Also put this repo folder under [tensorflow](https://github.com/tensorflow/tensorflow) root folder.

The Android entries in ```<workspace_root>/WORKSPAC``` must be uncommented with the paths filled in appropriately depending on where you installed the NDK and SDK, which is in [tensorflow](https://github.com/tensorflow/tensorflow) root folder. Otherwise an error such as: "The external label '//external:android/sdk' is not bound to anything" will be reported.

Create ```asset``` folder, put ```graph_train.pb``` and ```dict.txt``` inside it.

After that, run ```bazel build --copt='-std=c++11' //tensorflow/androidrnn:AndroidRNN```.

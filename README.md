# hw2-hw2_1-

For training make sure train,test = True,False in Main Execution Function at line 681
the bash command : /hw2_seq2seq2.sh training_data 

For testing make sure train,test = False, True in Main Execution Function at line 681
and  in def testmodel(arg)  ModelSaveLoc = "name of the folder of best model"
the bash coommand : /hw2_seq2seq2.sh testing_data output.txt
output.txt will have the generated captions. 
Also, keep the testing_label,json file in testing_data directetroy and training_label.json file in training_data directory.


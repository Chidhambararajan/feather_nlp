# feather_nlp
This project acts a upper layer for rasa nlu engine to make it work in low powered devices like Raspberry Pi

This project depends primarily on the Entity detection Engine of Rasa NLU and fuzzy wuzzy string search techniques

It doesnt employ any machine learning embeddings provided by rasa nlu , as a result this engine in certain test cases 
demands more training data than rasa nlu engine with machine learning embeddings .

This project is focussed and developed to encourage the deployment of NLP (Natural Language Processing) as an offline tool 
for low powered devices like Raspberry Pi (yet the project will work in Linux and Unix based desktop and server distros too..)


#**Sample Code**
```
import feather_nlp

feather_nlp.train('nlp.md','sample')
# "nlp.md" is the training data file in Markdown format
# "sample" is a folder yet to be created in which the trained data will be stored.
# incase the folder exists it will give a warning and erase the folder for further proceedings

feather_nlp.train('nlp.md','sample',no_warning=True)
#does the same like the above method , doesnt give any warning 

feather_engine=feather_nlp.feather_interpreter('trained_data_directory')

feather_engine.parse('<your test string here>')
#The above method will return the output in python dictionary format
 ```

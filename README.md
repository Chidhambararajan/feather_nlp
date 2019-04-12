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

#===========Training Area==========================================================

feather_nlp.train('nlp.md','sample',no_warning=True)
#does the same like the above method , doesnt give any warning 

feather_engine=feather_nlp.feather_interpreter('trained_data_directory')

#============Accessing Trained Data================================================

feather_engine.parse(str(input('Enter the inp query :')))
#The above method takes a string a input and returns the output in python dictionary data type
 ```

**Sample Output**
```
Enter the input query : who are the children of Indhira Gandhi
{'entities': [{'confidence': 0.9990499730270404,
               'end': 20,
               'entity': 'attribute',
               'extractor': 'ner_crf',
               'start': 12,
               'value': 'children'},
              {'confidence': 0.9993391254402305,
               'end': 38,
               'entity': 'subject',
               'extractor': 'ner_crf',
               'start': 24,
               'value': 'Indhira Gandhi'}],
 'intent': {'confidence': 0.9991945492336354, 'value': 'internet-skill'}}

```

For more efficient offline NLP engine one can go for [Rasa NLU engine](https://rasa.com/docs/nlu/)

Give a star if you like the project !!!

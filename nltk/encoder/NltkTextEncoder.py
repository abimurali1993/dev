import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing import text
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.stem import WordNetLemmatizer 
import pandas as pd
import re

class NltkTextEncoder:

    """
    Natural Language Tool Kit Based Encoder
    
    Given a series of text, the encoder encodes the text to form a dataframe of text vectors and has the capability to perform 
    WordNet Lematizer, padding, remove stopwords, HTML tags, email addresses, numericals and custom regualr expressions. 
    
    Parameters
    ----------
    :param dataFeature: Input text field data either as series of dataFrame colum or list of text
    :param language: language to set stopwords to be removed.(default is 'english')
    :param maximumWords: maximum words taken for consideration based on the frequency of the words.(default set to 10000)
    :param maximumLength: maximum lengtth taken for padding.(default set to 25)
    :param padding: poadding can be set either to 'pre' or 'post'.(default set to 'post')
    :param paddingValue: Value set to pad the vectors. (default set to 0)
    :param getWordIndex: returns the word index dictionary.(default set to False)
    :param wordNetLematizer: will perform wordNetLematization on the text.(default set to True)
    :param removeNumericals: removes Numericals in the text.(default set to True)
    :param removeHTML: removes HTML tags and email adresses in the text.(default set to True)
    :param removeCustom: flag for user to remove any custom text using regular expressions.(default set to False)
    :param customExpression: regular expression for user to define to remove from text
    
    Type
    ----------
    :type dataFeature: List and DataFrame
    :type language: str
    :type maximumWords: int
    :type maximumLength: int
    :type padding: str
    :type paddingValue: int
    :type getWordIndex: Boolean
    :type wordNetLematizer: Boolean
    :type removeNumericals: Boolean
    :type removeHTML: Boolean
    :type removeCustom: Boolean
    :type customExpression: str
    
    Returns
    ----------
    :returns: a dataFrame of encoded text vectors and word index
    
    Example
    ----------
    Input:
        
    data['headline'][0]='This is a beautiful/ <day?'
    data['headline'][1]='There are differences in time zones96'
    0    This is a beautiful/ <day?
    1    There are differences in time zones96
    Output:
        
    (   0  1  2  3  4
    0  1  2  0  0  0
    1  3  4  5  0  0, {'beautiful': 1, 'day': 2, 'difference': 3, 'time': 4, 'zone': 5})
    
    
    @author: Abilash Cheruvathur <abi.murali2009@gmail.com>
    """
    
    def __init__(self,language='english',maximumWords=10000,maxLength=25,padding='post',paddingValue=0,getWordIndex=False, 
                 wordNetLematizer=True, removeNumericals=True, removeHTML=True, removeCustom=False, customExpression=' '):
        self.language = language
        self.maximumWords = maximumWords
        self.maxLength = maxLength
        self.padding = padding
        self.paddingValue = paddingValue
        self.getWordIndex = getWordIndex
        self.wordNetLematizer = wordNetLematizer
        self.removeNumericals = removeNumericals
        self.removeHTML = removeHTML
        self.removeCustom = removeCustom
        self.customExpression = customExpression

    """
    Function to remove HTML tags and email address
    :param: record: text string
    :return:  text: text string
    """
    def __remove_HTML(self,record):
      cleaner = re.compile('<.*?>')
      text = re.sub(cleaner, ' ', record)
      text = re.sub(r'\S+@\S+', 'EmailId', text)
      text = re.sub(r'\'', '', text)
      return text
   
    """
    Function to remove numbers
    :param: record: text string
    :return:  text: text string
    """
    def __remove_numbers(self,record):
      text = re.sub(r'[0-9]', '', record)
      text = re.sub(r'[^a-zA-Z]', ' ', text)
      return text
    
    """
    Function to custom regular expressions
    :param: record: text string
    :return:  text: text string
    """
    def __remove_custom(self,record):
      cleaner = re.compile(self.customExpression)
      text = re.sub(cleaner, ' ', record)
      return text


    def fit_transform(self,dataFeature):
        
        nltk.download('wordnet')
        nltk.download('stopwords')
        nltk.download('punkt')
    
        
        #Language list supported by StopWords in nltk
        languageList=['arabic', 'azerbaijani', 'danish', 'dutch', 'english', 'finnish', 
                              'french', 'german', 'greek', 'hungarian', 'indonesian', 'italian', 
                              'kazakh', 'nepali', 'norwegian', 'portuguese', 'romanian', 'russian', 
                            'slovene', 'spanish', 'swedish', 'tajik', 'turkish']
        
        #To Store the filtered tokens
        filteredTokens=[]
        
        #Initialize the Lemmatizer
        lemmatizer = WordNetLemmatizer()
        
        #Checking if the language passed is available in the language list supported by stopwords
        if self.language in languageList:
          stopWords=set(stopwords.words(self.language))
        else:
          raise ValueError("Language not supported by nltk stopwords")
        
        #Check if the dataFeature passed is not None type
        if dataFeature is not None:
          
          for index, record in enumerate(dataFeature):
            record = record.lower()
            
            #To Remove HTML tags and email addresses
            if self.removeHTML:
              record= self.__remove_HTML(record)
            else:
              pass
            
            #To remove numerical values
            if self.removeNumericals:
              record= self.__remove_numbers(record)
            else:
              pass
            
            #To remove custom regular expressions
            if self.removeCustom:
              record= self.__remove_custom(record)
            else:
              pass
            
            #WTokenize the words
            wordTokens = word_tokenize(record)
            filteredSentence=[word for word in wordTokens if not word in stopWords]
            
            #To perform wordNet Lematization on the tokens
            if self.wordNetLematizer:
              filteredSentence=[lemmatizer.lemmatize(word) for word in filteredSentence]
            else:
              pass
            
            filteredTokens.append(filteredSentence)
          
          #To fit the tokenizer
          if  type(self.maximumWords)==int and self.maximumWords>0:
            tokenizer = text.Tokenizer(num_words=self.maximumWords)
            tokenizer.fit_on_texts(filteredTokens)
            tokenizedFeatures = tokenizer.texts_to_sequences(filteredTokens)
          else:
              raise TypeError("maximumWords should be int and greater than 0")
          
        else:
          raise TypeError("None Type not supported")
    
        #To perform padding
        if self.padding in ['pre','post']:
          if type(self.maxLength)==int and self.maxLength>0:
            output=pad_sequences(maxlen=self.maxLength, sequences=tokenizedFeatures, padding=self.padding, value=self.paddingValue)
          else:
            raise TypeError("type of maxLength should be int and greater than 0")
    
        else:
          raise ValueError("padding should be either 'pre' or 'post'") 
        
        #To get the word index dictionary
        if self.getWordIndex:
          wordIndex=tokenizer.word_index
          return pd.DataFrame(output),wordIndex
        else:
          pass 
        
        return pd.DataFrame(output)

# -*- coding: utf-8 -*-
"""word_predictor.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1uR2UI4C0A7-OAe5K0ThcgQ_4QL4syidv
"""

faqs="""### **Caption for YouTube Video**
**"Unveiling Nature's Rainbow: The Stunning Beauty of Colorful Birds! 🌈🦜 | You Won't Believe Your Eyes!"**

### **Speech Script for the Video**

*(Opening Scene: A vibrant, colorful bird perched on a tree or flying in its natural habitat)*

**"Hey everyone, welcome back to our channel! In today’s video, we’re diving into one of nature’s most mesmerizing treasures: the colorful birds that paint the sky with their stunning hues. From the deep reds of the scarlet macaw to the electric blues of the kingfisher, we’re going to explore these feathered wonders up close and personal. 🌟"**

*(Showcase close-up shots of different birds, highlighting their vibrant feathers)*

**"Did you know that some of these birds use their dazzling colors to communicate with each other? Whether it’s for mating, warding off predators, or simply expressing their dominance, these birds are truly the superheroes of the animal kingdom. 🦸‍♂️🦸‍♀️"**

*(Cut to a bird in flight)*

**"But it’s not just their looks that make them extraordinary – many of these species have incredible adaptations that help them thrive in some of the most challenging environments on Earth. Imagine being able to navigate vast distances or blend in perfectly with your surroundings – that’s the true magic of these colorful birds!"**

*(Cut to birds interacting, singing, or in their natural environment)*

**"So, sit back, relax, and get ready to be amazed by the beauty and brilliance of nature. Don't forget to like, share, and subscribe if you want to see more stunning wildlife content! Hit that bell icon so you never miss a moment of wonder!"**

*(Ending Scene: Slow-motion footage of the bird flying away or perched beautifully in a tree)*

**"Thanks for watching, and remember – every bird you see is a piece of the natural world’s masterpiece. Until next time, keep exploring!"**

---

### **Key Points for Virality:**

1. **Emotion & Awe:** The speech should evoke awe and curiosity about the birds' beauty and abilities.
2. **Engagement:** Encourage viewers to like, comment, and share the video to increase engagement.
3. **Call to Action:** Remind viewers to subscribe and hit the bell icon for more content.
4. **Brevity & Energy:** Keep the script energetic and concise, allowing the visuals to take center stage while adding value through your narration.

This combination of a catchy title and an engaging, informative script will help maximize the video's potential to go viral!"""

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer=Tokenizer()

tokenizer.fit_on_texts([faqs])   #Convert text to word

tokenizer.word_index

len(tokenizer.word_index)

input_sequences=[]
for sentence in faqs.split('\n'):
  #print(sentence)
  tokenized_sentence=tokenizer.texts_to_sequences([sentence])[0]

  for i in range(1,len(tokenized_sentence)):
    input_sequences.append(tokenized_sentence[:i+1])

input_sequences

max_length=max([len(x) for x in input_sequences])

from tensorflow.keras.preprocessing.sequence import pad_sequences
padded_input_sequences=pad_sequences(input_sequences, maxlen=max_length,padding='pre')

"""In the below code we are removing the last column for the matrix to predict the word"""

padded_input_sequences

padded_input_sequences[:,:-1]

x=padded_input_sequences[:,:-1]

y=padded_input_sequences[:,-1]

y.shape

"""We will apply One Hot Encooding for the input example if the sentence is       
"Hi my name is ABC I live in xyx"
Total 9 words, to predict my  in the above sentence
 **[0 1 0 0 0 0 0 0 0]** in case we need

 if the output is [ 0.1 **0.6** 0.1 0.1 0.1 0.1 0.1 0.1 0.1] then we will take the second word because it has the highest probability
"""

from tensorflow.keras.utils import to_categorical
y=to_categorical(y, num_classes=237)



y.shape

y

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,LSTM,Dense

model=Sequential()
model.add(Embedding(237,100, input_length=56))
model.add(LSTM(150))
model.add(Dense(237,activation='softmax'))

x.shape

y.shape

model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])



model.summary()

model.fit(x,y,epochs=100)

import numpy
import time
text="Ending"

for i in range(10):

  #tokenize

  token_text=tokenizer.texts_to_sequences([text])[0]

  #padding
  padded_token_text = pad_sequences([token_text],maxlen=56,padding='pre')
  print(padded_token_text)

  model.predict(padded_token_text).shape
  position=np.argmax(model.predict(padded_token_text))

  for word,index in tokenizer.word_index.items():
    if index== position:
      text=text + " " + word
      print(text)
      time.sleep(1)



import numpy as np
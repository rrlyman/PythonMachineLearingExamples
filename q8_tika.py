#!/usr/bin/python
'''experiment with apache tika


@author: richard lyman
'''

import pytesseract 
import tika
from tika import  translate, detector, language


 
filename = '15-01-01 459_Mont_Lyman.jpg'
filename2 = 'img20150901_15233271bw.jpg'

from PIL import Image

rawText = pytesseract.image_to_string(Image.open(filename2), lang="rus")
print (rawText)
lines = rawText.split('\n')

import os
#os.putenv( 'TIKA_VERSION','default')  # - set to the version string, e.g., 1.12 or default to current Tika version.
#os.putenv( 'TIKA_SERVER_JAR','/home/richard/.m2/repository/org/apache/tika/tika-server/1.13/tika-server-1.13.jar') #- set to the full URL to the remote Tika server jar to download and cache.
os.putenv( 'TIKA_SERVER_ENDPOINT',' http://localhost:9998') #- set to the host (local or remote) for the running Tika server jar.
#os.putenv( 'TIKA_SERVER_ENDPOINT',' http://localhost:9998/language/string') #- set to the host (local or remote) for the running Tika server jar.
#os.putenv( 'TIKA_CLIENT_ONLY','True') #- if set to True, then TIKA_SERVER_JAR is ignored, and relies on the value for TIKA_SERVER_ENDPOINT and treats Tika like a REST client.
#os.putenv( 'TIKA_TRANSLATOR','org/apache/tika/language/translate/') #- set to the fully qualified class name (defaults to Lingo24) for the Tika translator implementation.
#os.putenv( 'TIKA_SERVER_CLASSPATH','/home/richard/.m2/repository/org/apache/tika/tika-server/1.13/tika-server-1.13.jar') #- set to a string (delimited by ':' for each additional path) to prepend to the Tika server jar path.
#os.putenv('TESSDATA_PREFIX','/usr/share/tesseract-ocr/4.00/tessdata/')
tika.initVM()
from tika import parser
parsed = parser.from_buffer("comme çi comme ça")
print(parsed["metadata"])
print(parsed["content"])
global Verbose
Verbose=True

result=translate.auto_from_buffer("comme çi comme ça", 'en')
print(result)
result = detector.from_buffer("comme çi comme ça")
print (result)
result = translate.from_buffer("comme çi comme ça",'fr','en')
print (result)
result = language.from_buffer("comme çi comme ça")
print (result)
for line in lines:
    if len(line)>0:
        result=translate.from_buffer(line, 'ru','en')
        print(result)

print ('\n########################### No Errors ####################################')
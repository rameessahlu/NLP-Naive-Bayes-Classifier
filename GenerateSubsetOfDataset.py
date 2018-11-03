import multiprocessing
import re
import json
import os

dataset_path = os.path.join('data', r'train.ft.txt')

def removeStopWords(ds):
    stop_words = set(['a','about','above','after','again','against','all','am','an','and','any','are','aren\'t','as','at','be','because','been','before','being','below','between','both','but','by','can\'t','cannot','could','couldn\'t','did','didn\'t','do','does','doesn\'t','doing','don\'t','down','during','each','few','for','from','further','had','hadn\'t','has','hasn\'t','have','haven\'t','having','he','he\'d','he\'ll','he\'s','her','here','here\'s','hers','herself','him','himself','his','how','how\'s','i','i\'d','i\'ll','i\'m','i\'ve','if','in','into','is','isn\'t','it','it\'s','its','itself','let\'s','me','more','most','mustn\'t','my','myself','no','nor','not','of','off','on','once','only','or','other','ought','our','ours', 'ourselves','out','over','own','same','shan\'t','she','she\'d','she\'ll','she\'s','should','shouldn\'t','so','some','such','than','that','that\'s','the','their','theirs','them','themselves','then','there','there\'s','these','they','they\'d','they\'ll','they\'re','they\'ve','this','those','through','to','too','under','until','up','very','was','wasn\'t','we','we\'d','we\'ll','we\'re','we\'ve','were','weren\'t','what','what\'s','when','when\'s','where','where\'s','which','while','who','who\'s','whom','why','why\'s','with','won\'t','would','wouldn\'t','you','you\'d','you\'ll','you\'re','you\'ve','your','yours','yourself','yourselves'])
    ds = re.sub(r'[\&\!\:\\\-\#\$\.\"\,\^\_\'\(\)]+', '', ds)
    ds_without_stop_words = [word for word in ds.split() if word not in stop_words]
    return ''.join(str(e) + " " for e in ds_without_stop_words)

def readBigFile(dataset_tpe, file_name, start_index):
	pool = multiprocessing.Pool(8)  # play around for performance
	ds_count = 0 
	#ngt_file = open(r'negative.txt', 'w')
	datasets = {}
	datasets["positive"] = []
	datasets["negative"] = []
	#with open(r"train.ft.txt") as f:
	#	if index is 10:
	#		return
	#	pool.map(do_stuff, f)
	#	index += 1
	with open(dataset_path, encoding="utf8") as f:
		for line in f:
			if ds_count >= start_index:
				if (len(datasets['positive']) + len(datasets['negative'])) < 1000:
					if ("__label__2" in line) and (len(datasets['positive']) <500):
						datasets["positive"] = datasets["positive"] + [removeStopWords(line[11:])]
					elif ("__label__1" in line) and (len(datasets['negative']) <500):
						#ngt_file.write(line[11:] + "\r\n")
						#ngt_wsw_file.write(removeStopWords(line[11:]) + "\r\n")
						datasets["negative"] = datasets["negative"] + [removeStopWords(line[11:])]
				else:
					#ngt_file.close()
					print(len(datasets['positive']))
					print(len(datasets['negative']))
					print(ds_count)
					with open( file_name, 'w') as outfile:
						json.dump(datasets, outfile)
					return
			ds_count = ds_count + 1

if __name__ == "__main__":
    print('Please enter 1 for generating finite train data 2 for test data:')
    choice = input()
    file_name = ''
    if int(choice,3) is 1:
        file_name = os.path.join('data', 'train data without stop words.json')
    else:
        file_name = os.path.join('data', 'test data without stop words.json')
    readBigFile(choice, file_name, 600)
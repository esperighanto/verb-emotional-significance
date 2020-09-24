import pandas as pd
import nltk
import matplotlib.colors as colors
import matplotlib.pyplot as pyplot

limit = 25
output_type = "text" # This can be changed to "graphed" as well.
entries = []
verb_count = []
importance = []
stressful = []
openness = []
draining = []
distracted = []

data = pd.read_csv('hippoCorpusV2.csv')

class Entry:
    def __init__(self, verbs, importance, stressful, openness, draining, distracted):
        self.verbs = verbs
        self.importance = importance
        self.stressful = stressful
        self.openness = openness
        self.draining = draining
        self.distracted = distracted

def verb_counter(tagged_text):
    verb_count = 0
    for i in range(len(tagged_text)):
        if(tagged_text[i][1] == "VB" or tagged_text[i][1] == "VBD" or tagged_text[i][1] == "VBG" or tagged_text[i][1] == "VBN" or tagged_text[i][1] == "VBP" or tagged_text[i][1] == "VBZ"):
            verb_count += 1
    return verb_count

for i, row in data.iterrows():
    # print(i, ": ", data['story'][i])

    text_to_check = nltk.word_tokenize(data['story'][i])
    tagged_text = nltk.pos_tag(text_to_check)

    entries.append(Entry(verb_counter(tagged_text), data['importance'][i], data['stressful'][i], data['openness'][i], data['draining'][i], data['distracted'][i]))
    if(output_type == "text"):
        print((i+1), "| Verb count:", entries[i].verbs, ", Importance:", entries[i].importance, ", Stressful:", entries[i].stressful, ", Openness:", entries[i].openness, ", Draining:", entries[i].draining, ", Draining:", entries[i].draining, ", Distracted:", entries[i].distracted)
    if(i >= limit-1):
        break

for i in range(len(entries)):
    verb_count.append(entries[i].verbs)
    importance.append(entries[i].importance)
    stressful.append(entries[i].stressful + 0.03)
    openness.append(entries[i].openness + 0.06)
    draining.append(entries[i].draining + 0.09)
    distracted.append(entries[i].distracted + 0.12)

# verb_count.sort()
importance.sort()
stressful.sort()
openness.sort()
draining.sort()
distracted.sort()

importance_averaged = []


pyplot.plot(importance, verb_count, c='#CC3360', label="Importance")
# pyplot.plot(stressful, verb_count, c='#FFFF00', label="Stressful")
# pyplot.plot(openness, verb_count, c='#FF9999', label="Openness")
# pyplot.plot(distracted, verb_count, c='#00CC66', label="Distracted")
# pyplot.plot(draining, verb_count, c='#0000CC', label="Draining")
pyplot.xlabel('Emotional Correlative Factors')
pyplot.ylabel('Verb Count')
pyplot.title('Emotional Correlative Factors As A Function Of Verb Count')
pyplot.legend()
pyplot.show()
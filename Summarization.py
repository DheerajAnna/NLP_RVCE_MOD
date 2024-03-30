import math
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

text = "The philosophy of Epicurus (341-270 B.C.E.) was a complete and interdependent system, involving a view of the goal of human life (happiness, resulting from absence of physical pain and mental disturbance), an empiricist theory of knowledge (sensations, together with the perception of pleasure and pain, are infallible criteria), a description of nature based on atomistic materialism, and a naturalistic account of evolution, from the formation of the world to the emergence of human societies. Epicurus believed that, on the basis of a radical materialism which dispensed with transcendent entities such as the Platonic Ideas or Forms, he could disprove the possibility of the soul’s survival after death, and hence the prospect of punishment in the afterlife. He regarded the unacknowledged fear of death and punishment as the primary cause of anxiety among human beings, and anxiety in turn as the source of extreme and irrational desires. The elimination of the fears and corresponding desires would leave people free to pursue the pleasures, both physical and mental, to which they are naturally drawn, and to enjoy the peace of mind that is consequent upon their regularly expected and achieved satisfaction. It remained to explain how irrational fears arose in the first place: hence the importance of an account of social evolution. Epicurus was aware that deeply ingrained habits of thought are not easily corrected, and thus he proposed various exercises to assist the novice. His system included advice on the proper attitude toward politics (avoid it where possible) and the gods (do not imagine that they concern themselves about human beings and their behavior), the role of sex (dubious), marriage (also dubious) and friendship (essential), reflections on the nature of various meteorological and planetary phenomena, about which it was best to keep an open mind in the absence of decisive verification, and explanations of such processes as gravity (that is, the tendency of objects to fall to the surface of the earth) and magnetism, which posed considerable challenges to the ingenuity of the earlier atomists. Although the overall structure of Epicureanism was designed to hang together and to serve its principal ethical goals, there was room for a great deal of intriguing philosophical argument concerning every aspect of the system, from the speed of atoms in a void to the origin of optical illusions."
words = word_tokenize(text)
sentences = sent_tokenize(text)
tf = {}
for word in words:
    if word not in tf:
        tf[word] = 1
    else:
        tf[word] += 1
idf = {}
N = 1 
for word in tf:
    idf[word] = math.log(N / tf[word])

tf_idf = {}
for word in tf:
    tf_idf[word] = tf[word] * idf[word]
sent_scores = {}
for sentence in sentences:
    sent_scores[sentence] = 0
    for word in word_tokenize(sentence):
        word = word.lower()
        if word in tf_idf:
            sent_scores[sentence] += tf_idf[word]

sorted_sent_scores = sorted(sent_scores.items(), key=lambda x: x[1], reverse=True)
print(sorted_sent_scores)
summary = []
for i in range(1):
    summary.append(sorted_sent_scores[i][0])
print("\n".join(summary))
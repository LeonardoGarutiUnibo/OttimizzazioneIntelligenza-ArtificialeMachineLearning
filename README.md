<h1>Elaborato soggetto ad esame per il corso "Laboratorio di Ottimizzazione, Intelligenza Artificiale e Machine Learning".</h1>


<h2>Descrizione del progetto</h2>

<p>
Il progetto propone di classificare automaticamente le emozioni espresse in testi brevi, utilizzando il dataset Emotion di HuggingFace. <br>
Le emozioni sono suddivise in 6 classi: joy, sadness, anger, fear, love, surprise.<br>
Il modello base utilizzato è RoBERTa-base, ulteriormente fine-tuned per il compito di classificazione multi-classe. <br>
Il progetto è sviluppato in PyTorch, sfruttando l’interfaccia Trainer di HuggingFace.<br>
</p>

<h2>Architettura</h2>
<p>
Modello: roberta-base (pretrained)<br>
Classificatore finale: adattato per 6 classi<br>
Loss Function: CrossEntropyLoss bilanciata secondo la distribuzione delle classi<br>
Trainer: HuggingFace Trainer<br>
</p>


<h2>Tecniche utilizzate:</h2>
<p>
Early stopping
Salvataggio automatico del miglior modello
Logging in locale
Confusion Matrix salvata come immagine
</p>

<h2>Dataset</h2>
<p>
Dataset: emotion<br>
Origine: HuggingFace Datasets<br>
Suddivisione: train, validation, test<br>
</p>

<h2>Analisi effettuata:</h2>
<p>
Distribuzione classi<br>
Bilanciamento tramite pesi nella loss<br>
</p>



<h3>Installare le dipendenze con:</h3>
pip install -r requirements.txt<br>


</h3>Esecuzione</h3>
Lancia l'addestramento con:<br>
python run.py<br>

<h3>Output finale</h3>
Modello addestrato salvato in final_model/<br>
Confusion Matrix in grafici/<br>
Report classificazione in console<br>

Es.

              precision    recall  f1-score   support
     sadness       0.95      0.97      0.96       581
         joy       0.96      0.93      0.95       695
        love       0.82      0.88      0.85       159
       anger       0.94      0.90      0.92       275
        fear       0.87      0.87      0.87       224
    surprise       0.70      0.80      0.75        66

    accuracy                           0.92      2000
   macro avg       0.87      0.89      0.88      2000
weighted avg       0.93      0.92      0.92      2000

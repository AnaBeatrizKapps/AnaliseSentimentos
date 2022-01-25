import pandas as pd
import os
import random
import spacy
from spacy.util import minibatch, compounding

def load_training_data(
    data_directory: str = "base/sentimentsTwitter-alterado.csv",
    split: float = 0.8,
    limit: int = 0
) -> tuple:
    # Load from files
    result = pd.read_csv ('base/sentimentsTwitter-alterado.csv')
    reviews = []
    for a in result['frase']:
      if type(a) != float:
        text = a 
        if a['sentimento' == 'positivo']:
          spacy_label = {
                          "cats": {
                          "positivo": "positivo" == True,
                          "negativo": "negativo" == False,
                          "neutro": "neutro" == False}
                        }
        elif a['sentimento' == 'negativo']:
          spacy_label = {
                          "cats": {
                          "positivo": "positivo" == False,
                          "negativo": "negativo" == True,
                          "neutro": "neutro" == False}
                        }
        else:
          spacy_label = {
                          "cats": {
                          "positivo": "positivo" == False,
                          "negativo": "negativo" == False,
                          "neutro": "neutro" == True}
                        }
        reviews.append((text, spacy_label))
    return reviews

def train_model(
    training_data: list,
    test_data: list,
    iterations: int = 4
) -> None:
    # Build pipeline
    nlp = spacy.load("pt_core_news_sm")
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.create_pipe(
            "textcat", config={"architecture": "simple_cnn"}
        )
        nlp.add_pipe(textcat, last=True)
    else:
        textcat = nlp.get_pipe("textcat")

    textcat.add_label("positivo")
    textcat.add_label("negativo")
    textcat.add_label("neutro")

    # Train only textcat
    training_excluded_pipes = [
        pipe for pipe in nlp.pipe_names if pipe != "textcat"
    ]
    with nlp.disable_pipes(training_excluded_pipes):
        optimizer = nlp.begin_training()
        # Training loop
        print("Beginning training")
        print("Loss\tPrecision\tRecall\tF-score")
        batch_sizes = compounding(
            4.0, 32.0, 1.001
        )  # A generator that yields infinite series of input numbers
        for i in range(iterations):
            print(f"Training iteration {i}")
            loss = {}
            random.shuffle(training_data)
            batches = minibatch(training_data, size=batch_sizes)
            for batch in batches:
                text, labels = zip(*batch)
                nlp.update(text, labels, drop=0.2, sgd=optimizer, losses=loss)
            with textcat.model.use_params(optimizer.averages):
                evaluation_results = evaluate_model(
                    tokenizer=nlp.tokenizer,
                    textcat=textcat,
                    test_data=test_data
                )
                print(
                    f"{loss['textcat']}\t{evaluation_results['precision']}"
                    f"\t{evaluation_results['recall']}"
                    f"\t{evaluation_results['f-score']}"
                )

    # Save model
    with nlp.use_params(optimizer.averages):
        nlp.to_disk("model_artifacts")

def evaluate_model(
    tokenizer, textcat, test_data: list
) -> dict:
    reviews, labels = zip(*test_data)
    reviews = (tokenizer(review) for review in reviews)
    true_positives = 0
    false_positives = 1e-8  # Can't be 0 because of presence in denominator
    true_negatives = 0
    false_negatives = 1e-8
    for i, review in enumerate(textcat.pipe(reviews)):
        true_label = labels[i]["cats"]
        for predicted_label, score in review.cats.items():
            # print("label:")
            # print(true_label)
            # print("score:")
            # print(score)
            # Every cats dictionary includes both labels. You can get all
            # the info you need with just the pos label.
            if (
                predicted_label == "negativo"
            ):
                continue
            if score >= 0.5 and true_label["positivo"]:  
                true_positives += 1
            elif score >= 0.5 and true_label["negativo"]:
                false_positives += 1
            elif score < 0.5 and true_label["negativo"]:
                true_negatives += 1
            elif score < 0.5 and true_label["positivo"]:
                false_negatives += 1
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    if precision + recall == 0:
        f_score = 0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    return {"precision": precision, "recall": recall, "f-score": f_score}

def test_model(TEST_REVIEW):
    input_data: str = TEST_REVIEW
    #  Load saved trained model
    loaded_model = spacy.load("model_artifacts")
    # Generate prediction
    parsed_text = loaded_model(input_data)
    # Determine prediction to return
    if parsed_text.cats["positivo"] > parsed_text.cats["negativo"]:
        prediction = "Positive"
        score = parsed_text.cats["positivo"]
    else:
        prediction = "Negative"
        score = parsed_text.cats["negativo"]
    print(
        f"Review text: {input_data}\nPredicted sentiment: {prediction}"
        f"\tScore: {score}"
    )

TEST_REVIEW = """
Profissionais educados, atenciosos e o mais importante, o problema foi resolvido, aos profissionais envolvidos, meus parabéns e obrigado.
"""
    

train_model(load_training_data(limit=100), load_training_data(limit=100))
test_model(TEST_REVIEW)
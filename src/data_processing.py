import json


def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)['data']
        items = []
        for article in data:
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    question = qa['question']
                    qid = qa['id']
                    answers = qa['answers'] if qa['answers'] else [{'text': ''}]
                    answerable = 'yes' if qa['answers'] else 'no'
                    answer_start = int(answers[0]['answer_start']) if qa['answers'] else -1
                    answer_end = answer_start + len(answers[0]['text']) if qa['answers'] else -1
                    items.append({
                        'context': context,
                        'question': question,
                        'answer': answers[0]['text'],
                        'id': qid,
                        'answerable': answerable,
                        'answer_start': answer_start,
                        'answer_end': answer_end
                    })
        return items

# Import flask and datetime module for showing date and time
from flask import Flask, request
import datetime
import pandas as pd

x = datetime.datetime.now()
  
# Initializing flask app
app = Flask(__name__)
  
# Route for seeing a data
@app.route('/data')
def get_time():
  
    # Returning an api for showing in  reactjs
    return {
        'Name':"geek", 
        "Age":"22",
        "Date":x, 
        "programming":"python"
        }

#return csv file three columns data as arrays
@app.route('/comments2')
def getdata():
    df2 = pd.read_csv(r"comments.csv")

    sentiarray = df2['sentiment_predictions'].tolist()
    sarcarray = df2['sarcasm_predictions'].tolist()
    commarray = df2['rawcomment'].tolist()
    
    return{
        'Sentiarray' : sentiarray,
        'Sarcarray' : sarcarray,
        'Commarray' : commarray
    }

#return csv file three columns data count (did for test🫤)
@app.route('/comments')
def get_data2():
    df = pd.read_csv(r"comments.csv")
    senti_count = df['sentiment_predictions'].count()
    sarc_count = df['sarcasm_predictions'].count()
    comment_count = df['rawcomment'].count()
    all_count = str(senti_count) + " " + str(sarc_count) + " " + str(comment_count)


    return {'All_count' : all_count,
            }
  

# ////////// get data from front-end (not working 🥲🤔)

@app.route('/api/process-input', methods=['GET'])
def process_input():
    input_data = request.get_json()['input']
    # Perform processing on input_data
    return {'output': 'some output'}

#///////////

# Running app
if __name__ == '__main__':
    app.run(debug=True)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from flask import Flask, render_template, request, url_for, send_from_directory, flash, redirect
from detector.classifer import CoinDetector, Coin

MAIN_PATH = os.path.abspath(os.path.dirname(__file__))
PREDICTED_PREFIX= "p_"

# Init Flask app
app = Flask(__name__, static_url_path='/static', )
app.secret_key = "fjfjdfoo393kd009d2-o2o"

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg',])


# Init Coin Detector
def save_predicted_coins(input_file, output_file):
    detector = CoinDetector(input_file)

    coins = []
    for title, v in [('One', 1), ('Two', 2), ('Five', 5), ('Ten', 10)]:
        coins.append(Coin(title, str(v), v, str(v)))

    detector.upload_coins(coins=coins, folder=os.path.join(MAIN_PATH, "detector", "examples"))
    detector.draw_predictions()
    detector.save(output_file)

    return detector


# Views
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('image')

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            return redirect(url_for('detector', filename=filename))

    return render_template('index.html')


@app.route('/detector/<filename>', methods=['GET'])
def detector(filename):
    d = save_predicted_coins(
        input_file=os.path.join(app.config['UPLOAD_FOLDER'], filename),
        output_file=os.path.join(app.config['UPLOAD_FOLDER'], PREDICTED_PREFIX + filename)
    )

    return render_template(
        'detector.html',
        main_image=filename,
        predicted_image=PREDICTED_PREFIX + filename,
        total_value=d.total_value,
        count_coins =d.count_coins
    )


@app.route('/uploaded/<filename>')
def uploaded(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)

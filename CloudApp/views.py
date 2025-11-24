from django.shortcuts import render
from django.db import connection
from datetime import datetime
from django.contrib import messages
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import os

# Defer heavy ML and data imports until they're actually needed. This avoids
# import-time failures when running management commands (check/migrate) on
# systems that don't have optional ML dependencies installed.
_ml_initialized = False
_ml_model = None

def _load_ml_model():
    """Lazily import ML libs, train the model, and cache it.

    Returns a callable `predict(data)`-like object (here a fitted estimator).
    If imports fail, returns None and leaves an informative message in logs.
    """
    global _ml_initialized, _ml_model
    if _ml_initialized:
        return _ml_model
    _ml_initialized = True
    try:
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from Homomorphic import encryptData
    except Exception:
        # Optional ML dependencies not available; don't block Django management commands.
        _ml_model = None
        return None

    # Dataset path: repo root / Dataset/heart.csv
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Dataset", "heart.csv")
    try:
        dataset = pd.read_csv(dataset_path)
    except Exception:
        _ml_model = None
        return None

    data = dataset.values
    X = data[:, 0 : data.shape[1] - 1]
    Y = data[:, data.shape[1] - 1]
    homo_X = encryptData(X)
    X_train, X_test, y_train, y_test = train_test_split(homo_X, Y, test_size=0.2)
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    _ml_model = (rf, encryptData)
    return _ml_model

def UploadCloudAction(request):
    if request.method == 'POST':
        global username
        # Read form inputs (keep as plain Python types)
        age = request.POST.get('t1', False)
        gender = request.POST.get('t2', False)
        cp = request.POST.get('t3', False)
        blood = request.POST.get('t4', False)
        chol = request.POST.get('t5', False)
        fbs = request.POST.get('t6', False)
        ecg = request.POST.get('t7', False)
        thalac = request.POST.get('t8', False)
        exang = request.POST.get('t9', False)
        peak = request.POST.get('t10', False)
        slope = request.POST.get('t11', False)
        ca = request.POST.get('t12', False)
        thal = request.POST.get('t13', False)

        # Keep a canonical plaintext representation to store alongside any encrypted data
        plaintext_vals = [age, gender, cp, blood, chol, fbs, ecg, thalac, exang, peak, slope, ca, thal]
        plaintext_str = " ".join(str(x) for x in plaintext_vals)

        # Try to load the ML model and perform encryption/prediction. If unavailable,
        # fall back to storing plaintext and a helpful prediction message.
        model = _load_ml_model()
        output = "Prediction unavailable"
        encrypted_str = ""
        if model is not None:
            try:
                import numpy as np
                rf, encrypt_func = model
                data = np.asarray([ [int(age), int(gender), int(cp), int(blood), int(chol), int(fbs), int(ecg), int(thalac), int(exang), float(peak), int(slope), int(ca), int(thal)] ])
                enc_data = encrypt_func(data)
                predict = rf.predict(enc_data)
                output = "Normal"
                # original logic treated predict==1 as Abnormal
                try:
                    if int(predict[0]) == 1:
                        output = "Abnormal"
                except Exception:
                    # non-integer/shape issues — keep generic output
                    pass

                # Build encrypted string (space-separated values) and append plaintext part
                enc_parts = []
                for i in range(len(enc_data)):
                    enc_parts.append(" ".join(str(x) for x in enc_data[i]))
                encrypted_str = ",".join(enc_parts) + "," + plaintext_str
            except Exception:
                # If prediction/encryption fails, fall back to plaintext storage
                encrypted_str = ",".join(["", plaintext_str])
                output = "Prediction failed"
        else:
            # ML not available; store empty encrypted part and plaintext after comma
            encrypted_str = ",".join(["", plaintext_str])

        today = str(datetime.now())
        with connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO patientdata(username, patient_data, predict, predict_date) VALUES (%s, %s, %s, %s)",
                [username, encrypted_str, output, today],
            )

        encrypted = encrypted_str.split(",")
        # Ensure we always have at least the encrypted (maybe empty) and plaintext parts
        enc_display = encrypted[0] if len(encrypted) > 0 else ""
        result = "Encrypted Data = "+enc_display+"<br/>Predicted Patient Health : "+output
        context= {'data':result}
        return render(request,'PatientScreen.html', context)

def ViewPrediction(request):
    if request.method == 'GET':
        global username
        output = '<table border=1 align=center>'
        output+='<tr><th><font size=3 color=black>Patient Name</font></th>'
        output+='<th><font size=3 color=black>Encrypted Symptoms</font></th>'
        output+='<th><font size=3 color=black>Decrypted Symptoms</font></th>'
        output+='<th><font size=3 color=black>Predicted Health</font></th>'
        output+='<th><font size=3 color=black>Date</font></th></tr>'
        with connection.cursor() as cursor:
            cursor.execute("select * from patientdata where username=%s", [username])
            lists = cursor.fetchall()
            for ls in lists:
                enc = ls[1].split(",")
                output += '<tr><td><font size=3 color=black>' + str(ls[0]) + '</font></td>'
                output += '<td><font size=3 color=black>' + enc[0] + '</font></td>'
                output += '<td><font size=3 color=black>' + enc[1] + '</font></td>'
                output += '<td><font size=3 color=black>' + ls[2] + '</font></td>'
                output += '<td><font size=3 color=black>' + ls[3] + '</font></td></tr>'
        context= {'data':output}        
        return render(request,'PatientScreen.html', context)

def PatientData(request):
    if request.method == 'GET':
        global username
        output = '<table border=1 align=center>'
        output+='<tr><th><font size=3 color=black>Patient Name</font></th>'
        output+='<th><font size=3 color=black>Encrypted Symptoms</font></th>'
        output+='<th><font size=3 color=black>Decrypted Symptoms</font></th>'
        output+='<th><font size=3 color=black>Predicted Health</font></th>'
        output+='<th><font size=3 color=black>Date</font></th></tr>'
        with connection.cursor() as cursor:
            cursor.execute("select * from patientdata")
            lists = cursor.fetchall()
            for ls in lists:
                enc = ls[1].split(",")
                output += '<tr><td><font size=3 color=black>' + str(ls[0]) + '</font></td>'
                output += '<td><font size=3 color=black>' + enc[0] + '</font></td>'
                output += '<td><font size=3 color=black>' + enc[1] + '</font></td>'
                output += '<td><font size=3 color=black>' + ls[2] + '</font></td>'
                output += '<td><font size=3 color=black>' + ls[3] + '</font></td></tr>'
        context= {'data':output}        
        return render(request,'DoctorScreen.html', context)     

def UploadCloud(request):
    if request.method == 'GET':
        output = '<tr><td><font size="3" color="black">Age</td><td><select name="t1">'
        for i in range(10, 100):
            output += '<option value="'+str(i)+'">'+str(i)+'</option>'
        output += '</select></td></tr>'
        context= {'data1':output}
        return render(request,'UploadCloud.html', context)

def index(request):
    if request.method == 'GET':
        return render(request,'index.html', {})

def Register(request):
    if request.method == 'GET':
       return render(request, 'Register.html', {})
    
def DoctorLogin(request):
    if request.method == 'GET':
       return render(request, 'DoctorLogin.html', {})

def PatientLogin(request):
    if request.method == 'GET':
       return render(request, 'PatientLogin.html', {})

def isUserExists(username):
    is_user_exists = False
    with connection.cursor() as cursor:
        cursor.execute("select * from user_signup where username=%s", [username])
        lists = cursor.fetchall()
        for _ in lists:
            is_user_exists = True
    return is_user_exists

def RegisterAction(request):
    if request.method == 'POST':
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        email = request.POST.get('t4', False)
        address = request.POST.get('t5', False)
        desc = request.POST.get('t6', False)
        usertype = request.POST.get('t7', False)
        record = isUserExists(username)
        page = None
        if record == False:
            with connection.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO user_signup(username, password, phone_no, email, address, description, usertype) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                    [username, password, contact, email, address, desc, usertype],
                )
                # rowcount may not be available on all backends; attempt to give feedback
                data = "Signup Done! You can login now"
                context = {"data": data}
                return render(request, "Register.html", context)
        else:
            data = "Given "+username+" already exists"
            context= {'data':data}
            return render(request,'Register.html', context)


def checkUser(uname, password, utype):
    global username
    msg = "Invalid Login Details"
    with connection.cursor() as cursor:
        cursor.execute(
            "select * from user_signup where username=%s and password=%s and usertype=%s",
            [uname, password, utype],
        )
        lists = cursor.fetchall()
        for _ in lists:
            msg = "success"
            username = uname
            break
    return msg

def PatientLoginAction(request):
    if request.method == 'POST':
        global username
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        msg = checkUser(username, password, "Patient")
        if msg == "success":
            context= {'data':"Welcome "+username}
            return render(request,'PatientScreen.html', context)
        else:
            context= {'data':msg}
            return render(request,'PatientLogin.html', context)
        
def DoctorLoginAction(request):
    if request.method == 'POST':
        global username
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        msg = checkUser(username, password, "Doctor")
        if msg == "success":
            context= {'data':"Welcome "+username}
            return render(request,'DoctorScreen.html', context)
        else:
            context= {'data':msg}
            return render(request,'DoctorLogin.html', context)










        


        

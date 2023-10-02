from django.shortcuts import render
import base64
import easyocr
import re
import numpy as np
from PIL import Image

def extract(img):
    idno = ''
    name = ''
    fname = ''
    dob = ''
    gen = ''

    pflag = 0
    nflag = 0
    fflag = 0
    aflag = 0

    panno = re.compile('^[A-Z]{5,6}[0-9]{3,4}[A-Z]{1}')
    panname = re.compile('[A-Z]{2,25}(?:\s[A-Z]{2,25})')
    voterno = re.compile('^[A-Z]{3,4}[0-9]{6,7}')
    votername = re.compile('^[a-zA-Z]{2,25}\s[a-zA-Z]{2,15}\s[a-zA-Z]{2,15}')
    adno = re.compile('^[0-9]{4}\s[0-9]{4}\s[0-9]{4}$')
    adname = re.compile('^[A-Z]{1}[a-z]{1,25}\s[A-Z]{1}[a-z]{1,25}')
    dater = re.compile('\d{1,2}[-/]\d{1,2}[-/]\d{2,4}')
    gender = re.compile('(?:male|Male|MALE|female|Female|FEMALE)')
    namer = re.compile('(?:name|Name|NAME)')
    father = re.compile('(?:father|Father|FATHER)')
    elector = re.compile('(?:elector|Elector|ELECTOR)')

    reader = easyocr.Reader(['en'], gpu=False)

    result = reader.readtext(img, paragraph=False)

    for res in result:

        if adno.match(res[1]):
            idno = str(res[1])

            for r in result:

                if aflag == 0:
                    if re.search(adname, r[1]):
                        name = str(r[1])
                        aflag = 1

                if re.search(dater, r[1]):
                    temp = []
                    temp = re.findall(dater, str(r[1]))
                    if len(temp) > 0:
                        dob = temp[0]

                if re.search(gender, r[1]):
                    temp = []
                    temp = re.findall(gender, str(r[1]))
                    if len(temp) > 0:
                        gen = temp[0]
            break

        if panno.match(res[1]):
            idno = str(res[1])

            for r in result:

                if panno.match(r[1]):
                    pflag = 1

                if pflag == -1:
                    if re.search(panname, r[1]):
                        fname = str(r[1])
                        pflag = 0

                if pflag == 1:
                    if re.search(panname, r[1]):
                        name = str(r[1])
                        pflag = -1

                if re.search(dater, r[1]):
                    temp = []
                    temp = re.findall(dater, str(r[1]))
                    if len(temp) > 0:
                        dob = temp[0]

                if re.search(gender, r[1]):
                    temp = []
                    temp = re.findall(gender, str(r[1]))
                    if len(temp) > 0:
                        gen = temp[0]
            break

        if voterno.match(res[1]):
            pflag = 1
            idno = str(res[1])

            for r in result:

                if fflag == 1 and pflag == 0:
                    temp = []
                    if re.search(votername, r[1]):
                        temp.append(str(r[1]))
                    if re.search(panname, r[1]):
                        temp.append(str(r[1]))
                    if re.search(adname, r[1]):
                        temp.append(str(r[1]))
                    if len(temp) > 0:
                        fname = temp[0]
                    fflag = -1
                    pflag = -1

                if father.match(r[1]):
                    fflag = 1

                if nflag == 1 and pflag == 1:
                    temp = []
                    if re.search(votername, r[1]):
                        temp.append(str(r[1]))
                    if re.search(panname, r[1]):
                        temp.append(str(r[1]))
                    if re.search(adname, r[1]):
                        temp.append(str(r[1]))
                    if len(temp) > 0:
                        name = temp[0]
                    nflag = 0
                    pflag = 0

                if namer.match(r[1]) or elector.match(r[1]):
                    nflag = 1

                if re.search(dater, r[1]):
                    temp = []
                    temp = re.findall(dater, str(r[1]))
                    if len(temp) > 0:
                        dob = temp[0]

                if re.search(gender, r[1]):
                    temp = []
                    temp = re.findall(gender, str(r[1]))
                    if len(temp) > 0:
                        gen = temp[0]
            break

    return idno, name, fname, dob, gen


def home(request):
    if request.method == 'POST':
        try:
            image =request.FILES["imagedoc"]
            image_base =base64.b64encode(image.read()).decode("utf-8")
        except:
            return render(request, 'index.html')

        img = np.array(Image.open(image))

        id, name, fname, dob, gen = extract(img)

        return render(request, 'result.html', {'id': id, 'name': name, 'fname': fname, 'dob': dob, 'gen': gen, 'img': image_base})
    else:
        return render(request, 'index.html')


def evaluate(dataframe):
    true_blinks = deleteNonVisibleBlinks(convertAnnotationToBlinks(dataframe, 'blink_id'))
    pred_blinks = deleteNonVisibleBlinks(convertAnnotationToBlinks(dataframe, 'blink_id_pred'))
    return calculateResultsStatistics(true_blinks, pred_blinks)

def convertAnnotationToBlinks(annotations, blink_col):
    blinks = []
    index = 0
    while index < len(annotations.index):
        row = annotations.loc[index]
        if row[blink_col] > 0:
            id = row[blink_col]
            start = index
            notVisible = False
            if row['NV'] == True:
                notVisible = True
            while index < len(annotations.index) and row[blink_col] > 0:
                if row['NV'] == True:
                    notVisible = True
                index+=1
                row = annotations.loc[index]
            index -=1
            end = index
            blinks.append({'start': start, 'end':end, 'notVisible':notVisible})
        index+=1
    return blinks

def deleteNonVisibleBlinks(blinks):
    newBlinks = []
    for blink in blinks:
        if blink['notVisible'] == False :
            newBlinks.append(blink)
    return newBlinks

def calcFP(detectedBlinks, groundTruth):
    i=0
    j=0
    blinkFPCounter=0
    iou_detection=0.2
    while  i<len(detectedBlinks) or j<len(groundTruth):
        if i==len(detectedBlinks) and j<len(groundTruth):
            break
        if i<len(detectedBlinks) and j==len(groundTruth):
            blinkFPCounter+=1
            i+=1
            continue
        if iou(detectedBlinks[i],groundTruth[j])>iou_detection:
            i3=i
            iouArray3=[]
            k3=0
            while j<len(groundTruth) and i3<len(detectedBlinks):
                temp=iou(detectedBlinks[i3],groundTruth[j])
                if temp>iou_detection:
                    iouArray3.append(temp)
                    k3+=1
                    i3+=1
                else:
                    break
            if k3>1:
                max=iouArray3[0]
                index=0
                for f in range(1,k3):
                    if max<iouArray3[f]:
                        max=iouArray3[f]
                        index=f
                del detectedBlinks[i+index]
                del groundTruth[j]
                continue
            else:
                i+=1
                j+=1
                continue
        if detectedBlinks[i]['end']<groundTruth[j]['end']:
            blinkFPCounter+=1
            i+=1
        else:
            j+=1
    return blinkFPCounter

def iou(blink1,blink2):
    #intervals are 3 digits, middle one tells about the partial-0 / full-1 blink property
    min=blink1['start']
    diffMin=blink2['start']-blink1['start']
    if min>blink2['start']:
        diffMin=blink1['start']-blink2['start']
        min=blink2['start']
    max=blink1['end']
    diffMax=blink1['end']-blink2['end']
    if max<blink2['end']:
        diffMax=blink2['end']-blink1['end']
        max=blink2['end']
    unionCount=max-min-diffMin-diffMax+1
    if unionCount<=0:
        return 0
    return(unionCount/float(max-min+1))

def calculateResultsStatistics (pred_blinks, true_blinks):
    fp = calcFP(pred_blinks,true_blinks)
    fn = calcFP(true_blinks, pred_blinks)
    db = len(true_blinks)
    tp = db - fp

    precision = 0
    if db > 0:
        precision = tp/db

    recall = 0
    if((tp + fn)>0):
        recall = tp/(tp + fn)

    f1 = 0

    if(precision + recall) > 0:
        f1 = 2*precision*recall/(precision + recall)

    return f1, precision, recall, fp, fn, tp
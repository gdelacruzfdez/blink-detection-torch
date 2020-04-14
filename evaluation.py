
EYE_OPEN = 0
EYE_PARTIALLY_CLOSED = 1
EYE_CLOSED = 2

PARTIAL_BLINK = 0
COMPLETE_BLINK = 1

def evaluate(dataframe):
    true_blinks = deleteNonVisibleBlinks(convertAnnotationToBlinks(dataframe, 'blink_id'))
    print('POSITIVOS:', len(true_blinks))
    pred_blinks = deleteNonVisibleBlinks(convertAnnotationToBlinks(dataframe, 'blink_id_pred'))
    print('PREDICCIONES:', len(pred_blinks))
    return calculateResultsStatistics(true_blinks, pred_blinks)

def evaluatePartialBlinks(dataframe):
    true_blinks =  convertToIntervalsPartialComplete(dataframe,'blink_type')
    partial_true_blinks, complete_true_blinks =  extractPartialAndFullBlinks(true_blinks)
    print('LEN TRUE', len(partial_true_blinks), len(complete_true_blinks))
    pred_blinks =  convertToIntervalsPartialComplete(dataframe, 'blink_type_pred')
    partial_pred_blinks, complete_pred_blinks = extractPartialAndFullBlinks(pred_blinks)
    print('LEN PRED', len(partial_pred_blinks), len(complete_pred_blinks))
    return calculateResultsStatistics(complete_true_blinks, complete_pred_blinks)

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

def convertToIntervalsPartialComplete(annotations, blink_col, min_threshold=1):
  #the input is classic is frame part of blink or not.
  #1 is incomplete blink
  #2 is complete
  # min_threshold must be 0 (strict) or 1 (loose)
  # multi blink not divided
    blinks = []

    partial = False
    counter = 0
    startIndex = 0
    index = 0
    notVisible = False
    while index < len(annotations.index):
        row = annotations.loc[index]
        if EYE_PARTIALLY_CLOSED == row[blink_col]:
            if counter == 0:
                partial = True
                counter = 1
                startIndex = index
                if row['NV'] == True:
                    notVisible = True
            else:
                if partial:
                    counter+=1
                    if row['NV'] == True:
                        notVisible = True
                else:
                    if counter > min_threshold:
                        blinks.append({'start': startIndex, 'end': index - 1, 'blink_type': COMPLETE_BLINK, 'notVisible': notVisible})
                        notVisible = False
                    partial = True
                    counter = 1
                    startIndex = index
                    if row['NV'] == True:
                        notVisible = True
        elif EYE_CLOSED == row[blink_col]:
            if counter == 0:
                partial = False
                counter = 1
                startIndex = index
                if row['NV'] == True:
                    notVisible = True
            else:
                if not partial:
                    counter+=1
                    if row['NV'] == True:
                        notVisible = True
                else:
                    if counter > min_threshold:
                        blinks.append({'start': startIndex, 'end': index - 1, 'blink_type': PARTIAL_BLINK, 'notVisible': notVisible})
                        notVisible = False
                    partial = False
                    counter = 1
                    startIndex = index
                    if row['NV'] == True:
                        notVisible = True
        else:
            if counter > min_threshold:
                if partial:
                    blinks.append({'start': startIndex, 'end': index - 1, 'blink_type': PARTIAL_BLINK, 'notVisible': notVisible})
                else:
                    blinks.append({'start': startIndex, 'end': index - 1, 'blink_type': COMPLETE_BLINK, 'notVisible': notVisible})
            notVisible = False
            counter = 0
        
        index+=1
    
    return blinks
                    
def mergeNeighbourBlinks(blinks):
    results = []
    i = 1
    while i < len(blinks):
        if blinks[i-1]['end'] + 1 != blinks[i]['start']:
            results.append(blinks[i-1])
            i+=1
        else:
            blink = {'start': blinks[i-1]['start'], 'notVisible':blinks[i-1]['notVisible'] or blinks[i]['notVisible'], 'end': blinks[i]['end']}
            i+=1
            while i < len(blinks) and blink['end']+1 == blinks[i]['start']:
                blink = {'start': blinks[i-1]['start'],'notVisible':blinks[i-1]['notVisible'] or blinks[i]['notVisible'], 'end': blinks[i]['end']}
                i+=1
            blinks[i-1]=blink
        
        if i ==len(blinks):
            results.append(blinks[i-1])
    return results

def extractPartialAndFullBlinks(blinks):
    partial = []
    full = []
    for b in blinks:
        if PARTIAL_BLINK == b['blink_type']:
            partial.append(b)
        else:
            full.append(b)
    return deleteNonVisibleBlinks(mergeNeighbourBlinks(partial)), deleteNonVisibleBlinks(mergeNeighbourBlinks(full))



                    


      
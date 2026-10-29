import json, math

def response_format(ids):
    score_map={'type':'object','properties':{i:{'type':'number','minimum':-1,'maximum':1} for i in ids},'required':list(ids),'additionalProperties':False}
    return {'type':'json_schema','json_schema':{'name':'asset_scores','strict':True,'schema':{'type':'object','properties':{'scores':score_map},'required':['scores'],'additionalProperties':False}}}

def no_duplicates(pairs):
    out={}
    for k,v in pairs:
        if k in out:raise ValueError('duplicate JSON key: '+k)
        out[k]=v
    return out

def validate(content,ids):
    obj=json.loads(content,object_pairs_hook=no_duplicates)
    if not isinstance(obj,dict) or set(obj)!= {'scores'}:raise ValueError('expected scores object only')
    scores=obj['scores']
    if not isinstance(scores,dict) or set(scores)!=set(ids):raise ValueError('scores must cover exactly all supplied IDs')
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v)  for v in scores.values()):raise ValueError('score must be a finite number')
    # IDs were assigned in a pre-fixed, timestamp-specific shuffled asset order.
    # This tie rule is fixed, not selected using future returns.
    return sorted(ids,key=lambda i:(-scores[i],i))


def normalize_scores(scores):
    """Common positive rescaling; raw values remain the canonical ranking source.

    Sorting raw finite values avoids even theoretical floating-point tie collapse.
    """
    scale=max(1.0,max(abs(v) for v in scores.values()))
    return {k:v/scale for k,v in scores.items()},scale

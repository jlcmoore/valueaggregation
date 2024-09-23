import os
import urllib

QUESTION_STRINGS = ['one', 'two', 'three']
PROBABILITY_COLUMNS = [f'probability: {q_str}' for q_str in QUESTION_STRINGS]

def filename_to_variables(filename):
    filename = filename.replace('_', '&')
    filename = os.path.splitext(filename)[0]
    return urllib.parse.parse_qs(filename)

def explode_paginated_call(function, result_key, **kwargs):
    # For use in boto3
    nextToken = ''
    results = []
    
    if 'MaxResults' not in kwargs:
        kwargs['MaxResults'] = 100

    while nextToken is not None:
        if len(nextToken) > 0:
            result = function(NextToken=nextToken, **kwargs)
        else:
            result = function(**kwargs)

        if 'NextToken' in result:
            nextToken = result['NextToken']
        else:
            nextToken = None
        results += result[result_key]
    return results
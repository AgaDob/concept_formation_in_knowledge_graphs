import numpy as np
import pandas as pd


def stratified_sampling(data, new_data, frac=1, distribution='uniform'):
  """ Setting frac=1 results in no oversampling. 
  If frac >=1: all new triples added at least once + excess sampled
  If frac <1 : only samples triples added
  @param data: original, unaugmented data as a class object
  @param new_data: numpy array of new unique concept triples
  """ 

  print('Stratified sampling begins...')
  nb_triples_to_add = int(np.round(data.nb_entities * frac,0))

  # each triple will be added at least once, hence:
  nb_triples_to_sample = nb_triples_to_add - data.nb_entities

  nb_new_unique_triples = new_data.shape[0]
  new_triples_idx = [i for i in range(nb_new_unique_triples)]
  new_unique_triples = [tuple(new_data[i,:]) for i in range(nb_new_unique_triples)]


  ## Uniform Distribution
  if distribution == 'uniform':
    p = [1 for i in range(nb_new_unique_triples)]
    # normalise
    p = p/np.sum(p)


  ## Probability proportional to entity frequency
  if distribution == 'frequency':
    # Count # triples each ent participates in 
    counts = {ent:0 for ent in original_data.entity_set}
    for triple in data.train_triples:
      s = triple[0]
      o = triple[2]
      if s in counts.keys():
        counts[s]+=1
      if o in counts.keys():
        counts[o]+=1

    # Compute probability distribution
    p = []
    for triple in new_unique_triples:
      s = triple[0]
      p.append(counts[s])
    
    # normalise
    p = p/np.sum(p)



  ## Probability inversely proportional to entity frequency
  if distribution == 'inverse_frequency':
    
    # Count # triples each ent participates in 
    counts = {ent:0 for ent in original_data.entity_set}
    for triple in data.train_triples:
      s = triple[0]
      o = triple[2]
      if s in counts.keys():
        counts[s]+=1
      if o in counts.keys():
        counts[o]+=1

    # Compute probability distribution
    p = []
    for triple in new_unique_triples:
      s = triple[0]
      if counts[s]>0:
        p.append(1/counts[s])
      else:
        p.append(1)
    
    # normalise
    p = p/np.sum(p)

  percentage = nb_triples_to_add/(nb_triples_to_add + len(data.train_triples))*100
  print('Percentage of new triples in training data: {}%'.format(np.round(percentage,1)))

  # Pick triples to add to the dataset
  if frac >=1:
    indicies = np.random.choice(new_triples_idx, nb_triples_to_sample, p=p)
    oversampled_data = np.concatenate((new_data, new_data[indicies, :]), axis = 0)
  if frac < 1:
    indicies = np.random.choice(new_triples_idx, nb_triples_to_add, p=p)
    oversampled_data = new_data[indicies, :]

  return oversampled_data
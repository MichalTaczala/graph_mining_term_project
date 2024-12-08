##Final version of the code



##Imports

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import pandas as pd
import numpy as np

import networkx as nx



#Reading the files
def read_the_graph(file_name):
    data = pd.read_csv(file_name)
    return data



#Return the unique nodes

def get_unique_nodes(data):
  unique_nodes = set()
  for source in data["source"]:
    nodes = source[1:-1].split(',')
    if nodes == ['']:
      nodes = {}
    else:
      nodes ={int(n) for n in nodes}
    unique_nodes.update(nodes)

  for target in data["destination"]:
    nodes = target[1:-1].split(',')
    if nodes == ['']:
      nodes = {}
    else:
      nodes ={int(n) for n in nodes}
    unique_nodes.update(nodes)

  for id in data["missing node"]:
    unique_nodes.add(id)

  map_nodes = {node:k for k, node in enumerate(unique_nodes)}
  map_node_reverse = {k:node for node,k in map_nodes.items()}

  return unique_nodes, map_nodes,map_node_reverse




##Allow to map nodes id with idexs
def translate_to_map(map,list_i):
  return [map[elem] for elem in list_i]

#Function used to turn the string in the data into graph
def turn_to_set(string):
  nodes = string[1:-1].split(',')
  if nodes == ['']:
    nodes = {}
  else:
    nodes ={int(n) for n in nodes}
  return nodes

##Create the edge list of the graph
def get_final_node(data):
  unique_nodes,map_nodes,map_node_reverse = get_unique_nodes(data)

  node_feature = [map_node_reverse[i] for i in range(len(map_nodes))]
  sources = []
  destination = []
  labels = [-1 for _ in range(len(map_nodes))]

  number_of_reactions = len(data["source"])
  number_of_nodes = len(map_nodes)

  reaction_id_custom = number_of_nodes
  node_feature += [-1 for _ in range(number_of_reactions)]
  reac_train = {}
  for _,reaction in data.iterrows():

    source = turn_to_set(reaction["source"])
    result = turn_to_set(reaction["destination"])

    mapped_source = translate_to_map(map_nodes,source)
    mapped_result = translate_to_map(map_nodes,result)



    sources += mapped_source
    destination += [reaction_id_custom for _ in mapped_source]

    sources += [reaction_id_custom for _ in mapped_result]
    destination += mapped_result

    labels += translate_to_map(map_nodes,[reaction["missing node"]])

    reac_train[reaction_id_custom] = {"source": mapped_source + translate_to_map(map_nodes,[reaction["missing node"]]), "destination": mapped_result }

    reaction_id_custom += 1


  edge_index = [sources,destination]


  return node_feature,edge_index,labels,reac_train




##-------------------------------------------Graph creation------------------------------
#return the list of missing nodes
def get_missing_nodes(data):
  missing_nodes = []
  for _,reaction in data.iterrows():
    missing_node = int(reaction["missing node"])
    missing_nodes.append(missing_node)
  return missing_nodes

#retrive all reaction nodes
def get_reactions(all_nodes, non_reac_nodes):
  return [elem for elem in all_nodes if elem not in non_reac_nodes]

#nx graph creation -------------------------------------------------------------------------------------
#we test directed and then we will not directed
def graph_create(data):
    DG = nx.DiGraph()
    unique_nodes,map_nodes,map_node_reverse = get_unique_nodes(data)
    node_features,edge_index,labels,reac_train = get_final_node(data)

    missing_nodes = get_missing_nodes(data)


    all_nodes = np.union1d(np.array(edge_index[0]), np.array(edge_index[1]))
    all_nodes = np.union1d(all_nodes,missing_nodes)
    all_edges = [(edge_index[0][i],edge_index[1][i]) for i in range(len(edge_index[0]))]

    reaction_nodes = get_reactions(all_nodes, unique_nodes)

    #adding the nodes
    DG.add_nodes_from(unique_nodes,type='metabolite')
    DG.add_nodes_from(reaction_nodes,type='reaction')
    DG.add_edges_from(all_edges)

    #add missing metabolites (since no training is really useful)
    DG.add_edges_from([(missing_nodes[i],reaction_nodes[i]) for i in range(len(missing_nodes))])

    sources = set([elem for elem in edge_index[0] if elem not in reaction_nodes ])
    sources = sources.union(missing_nodes)
    return DG,sources,all_nodes

#____________________________________________________End of the graph creation

#_______________________________________________Adar Method__________________________________________________________________

## adar index method

#we will compute the adar on the undirected graph only between the reaction and the sources

def validate_adar(sources,DG,data,validation_query = 'dataset/Completion_valid_query.csv',validation_answer = 'dataset/Completion_valid_answer.csv'):
  node_features,edge_index,labels,reac_train = get_final_node(data)
  missing_nodes = get_missing_nodes(data)
  all_nodes = np.union1d(np.array(edge_index[0]), np.array(edge_index[1]))
  all_nodes = np.union1d(all_nodes,missing_nodes)
  
  
  valid_query = read_the_graph(validation_query)
  valid_answ = read_the_graph(validation_answer)
  reactions = {}
  reaction_id = len(all_nodes)

  final_result = {}

  for _,reaction in valid_query.iterrows():
    source = turn_to_set(reaction["source"])
    result = turn_to_set(reaction["destination"])

    reactions[reaction_id] = {'source' : source, "destination" : result}
    reaction_id +=1

  reaction_id = len(all_nodes)
  for missing_node in valid_answ['missing node']:

    reactions[reaction_id]["missing_node"] = missing_node
    reaction_id +=1


  #test part -----------------------------------------------------

  G = DG.to_undirected()
  

  for reaction_id,values in reactions.items():

    G.add_node(reaction_id,type='reaction')

    G.add_edges_from([(source,reaction_id) for source in values["source"]])
    G.add_edges_from([(desti,reaction_id) for desti in values["destination"]])

    # if len(values["source"]) != 0:
    #   e_bunch = [(s,sources_reac) for s in sources for sources_reac in values["source"] if s not in values["source"]]
    # else :
    #   e_bunch = [(s,sources_reac) for s in sources for sources_reac in values["destination"] if s not in values["destination"]]

    #print(e_bunch)
    e_bunch = list(set([(s,sources_reac) for s in sources for sources_reac in values["source"] if s not in values["source"]] + [(s,sources_reac) for s in sources for sources_reac in values["destination"] if s not in values["destination"]]))

    scores = nx.adamic_adar_index(G,e_bunch)



    final_scores = {src:score for (src, dst, score) in scores}
    #print(final_scores)



    best_10_scores = dict(sorted(final_scores.items(), key=lambda item: item[1],reverse = True)[:10])

    final_result[reaction_id] = best_10_scores

    G.remove_edges_from([(source,reaction_id) for source in values["source"]])
    G.remove_edges_from([(desti,reaction_id) for desti in values["destination"]])
    G.remove_node(reaction_id)

  number_of_correct_pred = 0
  number_of_true_pred = 0
  ranks = []  # To store ranks for MMR calculation
  for reaction_id, value in final_result.items():
        #print(reaction_id, ':', value)
        true_missing_node = reactions[reaction_id]["missing_node"]

        # Check if true missing node is in the top 10 predictions
        top_10_predictions = list(value.keys())
        if len(top_10_predictions) != 0 and top_10_predictions[0] == true_missing_node:
          number_of_true_pred +=1

        

        
        if true_missing_node in top_10_predictions:
            number_of_correct_pred += 1

            # Find the rank of the true missing node
            rank = top_10_predictions.index(true_missing_node) + 1  # 1-based index
            ranks.append(1 / rank)
        else:
            ranks.append(0)  # If not in top 10, reciprocal rank is 0

  hits_at_10 = number_of_correct_pred / len(final_result)
  mmr_at_10 = sum(ranks) / len(final_result)

  acc = number_of_true_pred / len(final_result)








  return final_result,acc,hits_at_10,mmr_at_10


#_____________________________Custom scoring method__________________________________________________________________________

def relation_node_scores(r1,r2,DG,alpha = 0.5,beta = 0.5):
  sources_r1 = r1["source"]
  sources_r2 = r2["source"]

  number_src_r2 = len(r2["source"])

  common_src = list(set(sources_r1) & set(sources_r2))
  num_common_src = len(common_src)

  dest_r1 = r1["destination"]
  dest_r2 = r2["destination"]

  number_dest_r2 = len(r2["destination"])

  common_dest = list(set(dest_r1) & set(dest_r2))

  num_common_dest = len(common_dest)

  uncommon_sources = list(set(common_src) ^ set(r2["source"]))

  uncommon_dest =  list(set(common_dest) ^ set(r2["destination"]))

  scores ={}
  for source in uncommon_sources:
    score =  alpha * num_common_src/max(number_src_r2,len(r1["source"])) + beta *num_common_dest/max(number_dest_r2,len(r1["destination"]))
    weigthed_score = score

    scores[source] = weigthed_score

  return scores




## Computes the model on the validation set
def validate_custom(DG,data,reac_train,sources,validation_query = 'dataset/Completion_valid_query.csv',validation_answer = 'dataset/Completion_valid_answer.csv'):
  
  _,edge_index,_,reac_train = get_final_node(data)
  missing_nodes = get_missing_nodes(data)
  all_nodes = np.union1d(np.array(edge_index[0]), np.array(edge_index[1]))
  all_nodes = np.union1d(all_nodes,missing_nodes)

  valid_query = read_the_graph(validation_query)
  if validation_answer != '':
    valid_answ = read_the_graph(validation_answer)
  reactions = {}
  reaction_id = len(all_nodes)

  final_result = {}

  for _,reaction in valid_query.iterrows():
    source = turn_to_set(reaction["source"])
    result = turn_to_set(reaction["destination"])

    reactions[reaction_id] = {'source' : source, "destination" : result}
    reaction_id +=1

  reaction_id = len(all_nodes)

  if validation_answer != '':
    for missing_node in valid_answ['missing node']:

      reactions[reaction_id]["missing_node"] = missing_node
      reaction_id +=1


  #test part -----------------------------------------------------

  #-------------------------------------Getting all connected Relations


  scores = {}
  i = 0
  for reaction_id,values in reactions.items():
    # if (i%10 == 0):
    #   print(i,'/',len(reactions))
    i+=1
    tested_relationships = set()
    for src in values["source"]:
      succ = set(DG.successors(src))
      tested_relationships.update(succ)
    for dest in values["destination"]:
      pred = set(DG.predecessors(dest))
      tested_relationships.update(pred)


    scores_temp = {node:0 for node in sources}

    for rel in tested_relationships:

      small_score = relation_node_scores(reactions[reaction_id],reac_train[rel],DG,0,1)
      for node,s in small_score.items():
        scores_temp[node] +=s
    scores[reaction_id] =dict(sorted(scores_temp.items(), key=lambda item: item[1],reverse = True)[:10])

  final_result = scores

  #--------------------evaluate
  acc= -1
  hits_at_10 = -1
  mmr_at_10 = -1

  if validation_answer != '':

    number_of_correct_pred = 0
    number_of_true_pred = 0
    ranks = []  # To store ranks for MMR calculation
    for reaction_id, value in final_result.items():
          #print(reaction_id, ':', value)
          true_missing_node = reactions[reaction_id]["missing_node"]

          # Check if true missing node is in the top 10 predictions
          top_10_predictions = list(value.keys())
          if len(top_10_predictions) != 0 and top_10_predictions[0] == true_missing_node:
            number_of_true_pred +=1

          # if len(top_10_predictions) == 0:
          #         print(reactions[reaction_id])
          if true_missing_node in top_10_predictions:
              number_of_correct_pred += 1

              # Find the rank of the true missing node
              rank = top_10_predictions.index(true_missing_node) + 1  # 1-based index
              ranks.append(1 / rank)
          else:
              ranks.append(0)  # If not in top 10, reciprocal rank is 0

    hits_at_10 = number_of_correct_pred / len(final_result)
    mmr_at_10 = sum(ranks) / len(final_result)

    acc = number_of_true_pred / len(final_result)
  

  return final_result,acc,hits_at_10,mmr_at_10,reactions







#GNNs models______________________________________________________________________________

def validate_model2(all_nodes,new_node_features,new_edge_index,model,validation_query='dataset/Completion_valid_query.csv',
                   validation_answer='dataset/Completion_valid_answer.csv',
                   batch_size=32):
    import torch
    from torch_geometric.data import Data
    import torch.nn.functional as F

    
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Read validation data
    
    valid_query = read_the_graph(validation_query)
    if validation_answer != '':
      valid_answ = read_the_graph(validation_answer)

    model.eval()

    reactions = {}
    reaction_id_start = len(all_nodes)
    reaction_id = reaction_id_start

    final_result = {}

    # Build the reactions dictionary
    for idx, reaction in valid_query.iterrows():
        source = turn_to_set(reaction["source"])
        result = turn_to_set(reaction["destination"])
        reactions[reaction_id] = {'source': source, 'destination': result}
        reaction_id += 1

    if validation_answer != '':
      reaction_id = reaction_id_start
      for idx, missing_node in enumerate(valid_answ['missing node']):
          reactions[reaction_id]["missing_node"] = int(missing_node)
          reaction_id += 1

    # Prepare labels
    if validation_answer != '':
      labels = [reactions[id]["missing_node"] for id in reactions.keys()]

    # Prepare node features and edge indices
    t_node_features = torch.tensor(new_node_features, dtype=torch.float).unsqueeze(1).to(device)
    t_edge_index = torch.tensor(new_edge_index, dtype=torch.long).to(device)

    # Prepare the data object
    data_valid = Data(x=t_node_features, edge_index=t_edge_index).to(device)

    # Build a mapping from edges to their indices
    edge_indices = {}
    for i in range(data_valid.edge_index.size(1)):
        src = int(data_valid.edge_index[0, i].item())
        tgt = int(data_valid.edge_index[1, i].item())
        edge_indices[(src, tgt)] = i

    # Prepare edge mask
    num_edges = data_valid.edge_index.size(1)
    original_edge_mask = torch.ones(num_edges, dtype=torch.bool).to(device)

    # Prepare validation data for batching
    reaction_items = list(reactions.items())
    if validation_answer != '':
      labels_list = labels

    # Initialize variables for metrics
    number_of_correct_pred = 0
    number_of_true_pred = 0
    ranks = []  # To store ranks for MMR calculation

    # Batch processing
    for batch_start in range(0, len(reaction_items), batch_size):
        batch_end = min(batch_start + batch_size, len(reaction_items))
        batch_items = reaction_items[batch_start:batch_end]
        if validation_answer != '':
         batch_labels = labels_list[batch_start:batch_end]

        # Collect data for the batch
        missing_nodes = []
        reaction_ids = []
        list1_batch = []
        list2_batch = []
        edges_to_remove = []
        new_node_indices = []
        batch_edge_index = data_valid.edge_index.clone()
        batch_node_features = data_valid.x.clone()
        batch_edge_mask = original_edge_mask.clone()

        # Keep track of current number of nodes and edges
        current_num_nodes = batch_node_features.size(0)
        current_num_edges = batch_edge_index.size(1)

        for idx_in_batch, (reaction_id, values) in enumerate(batch_items):
            if validation_answer != '':
             missing = batch_labels[idx_in_batch]
            if validation_answer != '':
              missing_nodes.append(missing)
            reaction_ids.append(reaction_id)

            # Assign a new node index for the reaction
            reaction_node_idx = current_num_nodes
            new_node_indices.append(reaction_node_idx)

            # Update node features
            reaction_node_feature = torch.tensor([[1.0]], dtype=torch.float).to(device)
            batch_node_features = torch.cat([batch_node_features, reaction_node_feature], dim=0)
            current_num_nodes += 1

            # Add edges from source to reaction node
            for src in values["source"]:
                edge = torch.tensor([[src], [reaction_node_idx]], dtype=torch.long).to(device)
                batch_edge_index = torch.cat([batch_edge_index, edge], dim=1)
                batch_edge_mask = torch.cat([batch_edge_mask, torch.tensor([True], dtype=torch.bool).to(device)], dim=0)
                current_num_edges += 1

                edge_tuple = (src, reaction_node_idx)
                edge_indices[edge_tuple] = current_num_edges - 1  # Update edge_indices

            # Add edges from reaction node to destination
            for dst in values["destination"]:
                edge = torch.tensor([[reaction_node_idx], [dst]], dtype=torch.long).to(device)
                batch_edge_index = torch.cat([batch_edge_index, edge], dim=1)
                batch_edge_mask = torch.cat([batch_edge_mask, torch.tensor([True], dtype=torch.bool).to(device)], dim=0)
                current_num_edges += 1

                edge_tuple = (reaction_node_idx, dst)
                edge_indices[edge_tuple] = current_num_edges - 1  # Update edge_indices

            # Prepare list1 and list2
            temp = list(values["source"])
            if validation_answer != '':
              if missing in temp:
                  temp.remove(missing)
            list1_batch.append(temp)
            list2_batch.append(list(values["destination"]))

            # Deactivate the edge between missing node and reaction node
            # edge_to_remove = (missing, reaction_node_idx)
            # edge_to_remove_id = edge_indices.get(edge_to_remove)
            # if edge_to_remove_id is not None:
            #     batch_edge_mask[edge_to_remove_id] = False
            #     edges_to_remove.append(edge_to_remove_id)

        # Prepare the data object for the batch
        batch_data = Data(
            x=batch_node_features,
            edge_index=batch_edge_index[:, batch_edge_mask],
        ).to(device)

        # Forward pass
        with torch.no_grad():
            out = model(batch_data.x, batch_data.edge_index, list1_batch=list1_batch, list2_batch=list2_batch)

        # Compute predictions
        _, indices_found = torch.topk(out, k=10, dim=1)  # Get top 10 predictions for each sample
        hits_at_10 = -1
        mmr_at_10 = -1
        acc= -1
        for idx_in_batch, reaction_id in enumerate(reaction_ids):
          # Store the final scores for the reaction
          top_10_predictions = indices_found[idx_in_batch].tolist()
          scores = out[idx_in_batch]
          class_scores = scores[top_10_predictions]
          final_scores = {top_10_predictions[i]: class_scores[i].item() for i in range(len(top_10_predictions))}
          final_result[reaction_id] = final_scores

        if validation_answer != '':
          # Evaluate predictions
          for idx_in_batch, reaction_id in enumerate(reaction_ids):
              true_missing_node = reactions[reaction_id]["missing_node"]
              top_10_predictions = indices_found[idx_in_batch].tolist()

              if len(top_10_predictions) != 0 and top_10_predictions[0] == true_missing_node:
                  number_of_true_pred += 1

              if true_missing_node in top_10_predictions:
                  number_of_correct_pred += 1
                  rank = top_10_predictions.index(true_missing_node) + 1  # 1-based index
                  ranks.append(1 / rank)
              else:
                  ranks.append(0)

              # Store the final scores for the reaction
              # scores = out[idx_in_batch]
              # class_scores = scores[top_10_predictions]
              # final_scores = {top_10_predictions[i]: class_scores[i].item() for i in range(len(top_10_predictions))}
              # final_result[reaction_id] = final_scores

          # No need to remove nodes and edges since we cloned the data for each batch

          # Calculate metrics
          total_samples = len(reactions)
          hits_at_10 = number_of_correct_pred / total_samples
          mmr_at_10 = sum(ranks) / total_samples
          acc = number_of_true_pred / total_samples

    if validation_answer != '':
      print(f"Validation Results - Accuracy: {acc:.4f}, Hits@10: {hits_at_10:.4f}, MMR@10: {mmr_at_10:.4f}")

    return final_result, acc, hits_at_10, mmr_at_10



###GAT Model ---------------------------------------------------------------------------------------------------------

def retrive_edges_list(DG):

  edge_list = list(DG.edges)

  edge_source =[i for i,_ in edge_list]
  edge_dest = [j for _,j in edge_list]

  return [edge_source,edge_dest]

def remove_edge(DG,source_node,dest_node):
  DG.remove_edge(source_node,dest_node)
  return DG

def GAT_model_crea(DG,data,reac_train,unique_nodes,all_nodes):
    import torch
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
    from torch_geometric.data import Data
    import torch.nn.functional as F
    import torch.nn as nn

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Assuming 'data' is your DataFrame containing reactions and 'DG' is your graph
    # Replace 'data' and 'DG' with your actual variables

    # Prepare labels
    missing_node = [int(reaction["missing node"]) for _, reaction in data.iterrows()]
    labels = missing_node

    reac_train_sampled = reac_train  # Assuming 'reac_train' is your training data

    #print(len(labels), '   ', len(reac_train_sampled))

    # Data preparation
    new_node_features = [0 for _ in unique_nodes]
    new_node_features += [1 for _ in reac_train]

    new_edge_index = retrive_edges_list(DG)

    # Convert inputs to PyTorch tensors and move to GPU
    t_node_features = torch.tensor(new_node_features, dtype=torch.float).unsqueeze(1).to(device)
    t_edge_index = torch.tensor(new_edge_index, dtype=torch.long).to(device)
    t_labels = torch.tensor(labels, dtype=torch.long).to(device)

    # Prepare PyTorch Geometric data and move to GPU
    data_ = Data(x=t_node_features, edge_index=t_edge_index, y=t_labels).to(device)

    # Define the EnhancedGCN model adjusted for batch processing
    class EnhancedGCN(nn.Module):
        def __init__(self, in_channels, num_classes, max_len_src, hidden_channels=64, num_heads=4, dropout=0.5):
            super().__init__()
            # GAT layers
            self.hidden_channels = hidden_channels
            self.max_len = max_len_src
            self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads, concat=True)
            self.gat2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True)
            self.gat3 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True)
            self.gat4 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=False)
        

            self.bn1 = nn.BatchNorm1d(hidden_channels * num_heads)
            self.bn2 = nn.BatchNorm1d(hidden_channels * num_heads)
            self.bn3 = nn.BatchNorm1d(hidden_channels * num_heads)
            self.bn4 = nn.BatchNorm1d(hidden_channels)
            

            # Linear layers for residual connections
            self.residual_proj1 = nn.Linear(hidden_channels * num_heads, hidden_channels * num_heads)
            self.residual_proj2 = nn.Linear(hidden_channels * num_heads, hidden_channels * num_heads)
            self.residual_proj3 = nn.Linear(hidden_channels * num_heads, hidden_channels)

            self.list_true_fc_1 = nn.Linear(hidden_channels*max_len_src, hidden_channels)
            self.list_true_fc_2 = nn.Linear(hidden_channels, hidden_channels)
        
            # Fully connected layers for processing the lists
            self.list_fc = nn.Linear(hidden_channels, hidden_channels)

            # Fully connected layers for classification
            self.fc1 = nn.Linear(hidden_channels * 2, hidden_channels*2)
            self.fc2 = nn.Linear(hidden_channels, num_classes)

            self.fc3 = nn.Linear(hidden_channels * 2, hidden_channels)


            # Dropout
            self.dropout = dropout

        def forward(self, x, edge_index, list1_batch, list2_batch, batch=None):
            # First GAT layer with dropout
            x = self.gat1(x, edge_index)
            x = self.bn1(x)
            x = F.relu(x)
            #x = F.dropout(x, p=self.dropout, training=self.training)


            # Second GAT layer with residual connection
            x_res1 = self.residual_proj1(x)
            x = self.gat2(x, edge_index)
            x = self.bn2(x)
            x = F.relu(x + x_res1)
            #x = F.dropout(x, p=self.dropout, training=self.training)

            x_res2 = x  # Residual connection
            x_res2 = self.residual_proj2(x_res2)  # Align dimensions for residual connection
            x = self.gat3(x, edge_index)
            x = self.bn3(x)
            x = F.relu(x + x_res2)  # Residual connection
            #x = F.dropout(x, p=self.dropout, training=self.training)

            x_res3 = self.residual_proj3(x)
            x = self.gat4(x, edge_index)
            x = self.bn4(x)
            x = F.relu(x + x_res3)
            #x = F.dropout(x, p=self.dropout, training=self.training)

        
            # Process embeddings from list1
            

            # combined_list1 = self.process_list(x, list1_batch)  # Shape: [batch_size, hidden_channels]

            # # Process embeddings from list2
            # combined_list2 = self.process_list(x, list2_batch)  # Shape: [batch_size, hidden_channels]
            
            combined_list1 = self.process_list(x, list1_batch)  # Shape: [batch_size, hidden_channels]

            # Process embeddings from list2
            combined_list2 = self.process_list(x, list2_batch)

            # Combine the two aggregated embeddings
            combined_embedding = torch.cat([combined_list1, combined_list2], dim=1)  # Shape: [batch_size, 2 * hidden_channels]

            # Fully connected layers for classification
            combined_embedding = F.relu(self.fc1(combined_embedding))  # Shape: [batch_size, hidden_channels]
            combined_embedding = F.dropout(combined_embedding, p=self.dropout, training=self.training)

            combined_embedding = F.relu(self.fc3(combined_embedding))
            combined_embedding = F.dropout(combined_embedding, p=self.dropout, training=self.training)
            
            logits = self.fc2(combined_embedding) # Shape: [batch_size, num_classes]

            return logits

        def process_list(self, x, node_lists):
            

            
            embeddings_list = []
            for node_list in node_lists:
                if len(node_list) > 0:
                    embeddings = x[node_list]  # Shape: [list_length, hidden_channels]
                    # Pad embeddings to max_len if necessary
                    pad_size = self.max_len - embeddings.size(0)
                    if pad_size > 0:
                        padding = torch.zeros(pad_size, embeddings.size(1), device=x.device)
                        embeddings_padded = torch.cat([embeddings, padding], dim=0)
                    else:
                        embeddings_padded = embeddings[:self.max_len]
                else:
                    embeddings_padded = torch.zeros(self.max_len, self.hidden_channels, device=x.device)
                embeddings_list.append(embeddings_padded)
            
            # Stack embeddings_list to get a tensor of shape [batch_size, max_len, hidden_channels]
            #batches = torch.stack(embeddings_list)
            #embeddings_padded_batch = batches.view(batches.shape[0],batches.shape[1]*batches.shape[2])  # Shape: [batch_size, max_len, hidden_channels]
            # Apply a fully connected layer to each node embedding
            #print(embeddings_padded_batch.shape)
            embeddings_padded_batch = torch.stack(embeddings_list)
            #combined_embeddings = self.list_true_fc_2(self.list_true_fc_1(embeddings_padded_batch))
            #print(combined_embeddings.shape)
            embeddings_transformed = F.relu(self.list_fc(embeddings_padded_batch))  # Shape: [batch_size, max_len, hidden_channels]
            # Aggregate embeddings
            combined_embeddings = embeddings_transformed.sum(dim=1)  # Shape: [batch_size, hidden_channels]
            return combined_embeddings

    # Initialize and move model to GPU
    max_len_src = max(
        max([len(values["source"]) for values in reac_train_sampled.values()]),
        max([len(values["destination"]) for values in reac_train_sampled.values()])
    )
    print(f"Max sequence length: {max_len_src}")

    model = EnhancedGCN(
        in_channels=data_.num_features,
        num_classes=len(unique_nodes),
        max_len_src=max_len_src
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # Prepare a mask tensor for edges
    num_edges = data_.edge_index.size(1)
    edge_mask = torch.ones(num_edges, dtype=torch.bool).to(device)  # All edges active initially

    # Modify the Data object to include the mask
    data_.edge_mask = edge_mask  # Adding edge mask to data

    # Build a mapping from edges to their indices
    edge_indices = {}
    for i in range(data_.edge_index.size(1)):
        src = int(data_.edge_index[0, i].item())
        tgt = int(data_.edge_index[1, i].item())
        edge_indices[(src, tgt)] = i

    # Training loop
    batch_size = 32  # Define your batch size
    num_epochs = 150  # Define the number of epochs

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        predicted_tokens = []

        # Prepare the data for batching
        reaction_items = list(reac_train_sampled.items())
        labels_list = labels  # Assuming labels is a list of missing nodes

        # Create batches
        for batch_start in range(0, len(reaction_items), batch_size):
            batch_end = min(batch_start + batch_size, len(reaction_items))
            batch_items = reaction_items[batch_start:batch_end]
            batch_labels = labels_list[batch_start:batch_end]

            # Collect missing nodes and reactions
            missing_nodes = []
            reaction_ids = []
            list1_batch = []
            list2_batch = []
            edges_to_remove = []

            for idx_in_batch, (reaction_id, values) in enumerate(batch_items):
                missing = batch_labels[idx_in_batch]
                reaction = reaction_id

                missing_nodes.append(missing)
                reaction_ids.append(reaction)

                # Prepare list1 and list2
                temp = values["source"].copy()
                if missing in temp:
                    temp.remove(missing)
                list1_batch.append(temp)
                list2_batch.append(list(values["destination"]))

                # Get edge to remove
                edge_to_remove = (missing, reaction)
                edge_to_remove_id = edge_indices.get(edge_to_remove)
                if edge_to_remove_id is not None:
                    edges_to_remove.append(edge_to_remove_id)

            # Deactivate edges
            data_.edge_mask[edges_to_remove] = False

            # Forward pass
            active_edges = data_.edge_index[:, data_.edge_mask]

            out = model(data_.x, active_edges, list1_batch=list1_batch, list2_batch=list2_batch)

            # Predicted tokens
            _, indices_found = torch.max(out, dim=1)  # indices_found: [batch_size]
            predicted_tokens.extend(indices_found.tolist())

            # Compute accuracy
            batch_labels_tensor = torch.tensor(missing_nodes, dtype=torch.long).to(device)
            correct += (indices_found == batch_labels_tensor).sum().item()

            # Compute loss and optimize
            optimizer.zero_grad()
            loss = criterion(out, batch_labels_tensor)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Reactivate edges
            data_.edge_mask[edges_to_remove] = True

        accuracy = correct / len(reac_train_sampled)
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")
        #print(f"Predicted tokens: {predicted_tokens}")
        if (epoch % 10 == 0):
            model.eval()
            final_result,acc,hits_at_10,mmr_at_10 = validate_model2(all_nodes,new_node_features,new_edge_index,model,validation_query='dataset/Completion_valid_query.csv',validation_answer='dataset/Completion_valid_answer.csv',batch_size=32)
    return model,new_edge_index,new_node_features
        


#Regular GCN -----------------------------------------------------------------------------------------------

def GCN_creation(reac_train,unique_nodes):
    import torch
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool,SAGEConv
    from torch_geometric.data import Data
    import torch.nn.functional as F
    import torch.nn as nn

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Assuming 'data' is your DataFrame containing reactions and 'DG' is your graph
    # Replace 'data' and 'DG' with your actual variables

    # Prepare labels
    missing_node = [int(reaction["missing node"]) for _, reaction in data.iterrows()]
    labels = missing_node

    reac_train_sampled = reac_train  # Assuming 'reac_train' is your training data

    print(len(labels), '   ', len(reac_train_sampled))

    # Data preparation
    new_node_features = [0 for _ in unique_nodes]
    new_node_features += [1 for _ in reac_train]

    new_edge_index = retrive_edges_list(DG)

    # Convert inputs to PyTorch tensors and move to GPU
    t_node_features = torch.tensor(new_node_features, dtype=torch.float).unsqueeze(1).to(device)
    t_edge_index = torch.tensor(new_edge_index, dtype=torch.long).to(device)
    t_labels = torch.tensor(labels, dtype=torch.long).to(device)

    # Prepare PyTorch Geometric data and move to GPU
    data_ = Data(x=t_node_features, edge_index=t_edge_index, y=t_labels).to(device)

    # Define the EnhancedGCN model adjusted for batch processing
    class EnhancedGCN(nn.Module):
        def __init__(self, in_channels, num_classes, max_len_src, hidden_channels=64, num_heads=4, dropout=0.5):
            super().__init__()
            # GAT layers
            self.hidden_channels = hidden_channels
            self.max_len = max_len_src
            self.gat1 = GCNConv(in_channels, hidden_channels)
            self.gat2 = GCNConv(hidden_channels, hidden_channels)
            self.gat3 = GCNConv(hidden_channels, hidden_channels)
            self.gat4 = GCNConv(hidden_channels, hidden_channels)
        

            self.bn1 = nn.BatchNorm1d(hidden_channels)
            self.bn2 = nn.BatchNorm1d(hidden_channels)
            self.bn3 = nn.BatchNorm1d(hidden_channels)
            self.bn4 = nn.BatchNorm1d(hidden_channels)
            

            # Linear layers for residual connections
            # self.residual_proj1 = nn.Linear(hidden_channels * num_heads, hidden_channels * num_heads)
            # self.residual_proj2 = nn.Linear(hidden_channels * num_heads, hidden_channels * num_heads)
            # self.residual_proj3 = nn.Linear(hidden_channels * num_heads, hidden_channels)

            self.list_true_fc_1 = nn.Linear(hidden_channels*max_len_src, hidden_channels)
            self.list_true_fc_2 = nn.Linear(hidden_channels, hidden_channels)
        
            # Fully connected layers for processing the lists
            self.list_fc = nn.Linear(hidden_channels, hidden_channels)

            # Fully connected layers for classification
            self.fc1 = nn.Linear(hidden_channels * 2, hidden_channels*2)
            self.fc2 = nn.Linear(hidden_channels, num_classes)

            self.fc3 = nn.Linear(hidden_channels * 2, hidden_channels)


            # Dropout
            self.dropout = dropout

        def forward(self, x, edge_index, list1_batch, list2_batch, batch=None):
            # First GAT layer with dropout
            x = self.gat1(x, edge_index)
            x = self.bn1(x)
            x = F.relu(x)
            #x = F.dropout(x, p=self.dropout, training=self.training)


            # Second GAT layer with residual connection
            x_res1 = x
            x = self.gat2(x, edge_index)
            x = self.bn2(x)
            x = F.relu(x + x_res1)
            #x = F.dropout(x, p=self.dropout, training=self.training)

            x_res2 = x  # Residual connection
            #x_res2 = self.residual_proj2(x_res2)  # Align dimensions for residual connection
            x = self.gat3(x, edge_index)
            x = self.bn3(x)
            x = F.relu(x + x_res2)  # Residual connection
            #x = F.dropout(x, p=self.dropout, training=self.training)

            x_res3 = x
            x = self.gat4(x, edge_index)
            x = self.bn4(x)
            x = F.relu(x + x_res3)
            #x = F.dropout(x, p=self.dropout, training=self.training)

        
            # Process embeddings from list1
            

            # combined_list1 = self.process_list(x, list1_batch)  # Shape: [batch_size, hidden_channels]

            # # Process embeddings from list2
            # combined_list2 = self.process_list(x, list2_batch)  # Shape: [batch_size, hidden_channels]
            
            combined_list1 = self.process_list(x, list1_batch)  # Shape: [batch_size, hidden_channels]

            # Process embeddings from list2
            combined_list2 = self.process_list(x, list2_batch)

            # Combine the two aggregated embeddings
            combined_embedding = torch.cat([combined_list1, combined_list2], dim=1)  # Shape: [batch_size, 2 * hidden_channels]

            # Fully connected layers for classification
            combined_embedding = F.relu(self.fc1(combined_embedding))  # Shape: [batch_size, hidden_channels]
            combined_embedding = F.dropout(combined_embedding, p=self.dropout, training=self.training)

            combined_embedding = F.relu(self.fc3(combined_embedding))
            combined_embedding = F.dropout(combined_embedding, p=self.dropout, training=self.training)
            
            logits = self.fc2(combined_embedding) # Shape: [batch_size, num_classes]

            return logits

        def process_list(self, x, node_lists):
            

            
            embeddings_list = []
            for node_list in node_lists:
                if len(node_list) > 0:
                    embeddings = x[node_list]  # Shape: [list_length, hidden_channels]
                    # Pad embeddings to max_len if necessary
                    pad_size = self.max_len - embeddings.size(0)
                    if pad_size > 0:
                        padding = torch.zeros(pad_size, embeddings.size(1), device=x.device)
                        embeddings_padded = torch.cat([embeddings, padding], dim=0)
                    else:
                        embeddings_padded = embeddings[:self.max_len]
                else:
                    embeddings_padded = torch.zeros(self.max_len, self.hidden_channels, device=x.device)
                embeddings_list.append(embeddings_padded)
            
            # Stack embeddings_list to get a tensor of shape [batch_size, max_len, hidden_channels]
            #batches = torch.stack(embeddings_list)
            #embeddings_padded_batch = batches.view(batches.shape[0],batches.shape[1]*batches.shape[2])  # Shape: [batch_size, max_len, hidden_channels]
            # Apply a fully connected layer to each node embedding
            #print(embeddings_padded_batch.shape)
            embeddings_padded_batch = torch.stack(embeddings_list)
            #combined_embeddings = self.list_true_fc_2(self.list_true_fc_1(embeddings_padded_batch))
            #print(combined_embeddings.shape)
            embeddings_transformed = F.relu(self.list_fc(embeddings_padded_batch))  # Shape: [batch_size, max_len, hidden_channels]
            # Aggregate embeddings
            combined_embeddings = embeddings_transformed.mean(dim=1)  # Shape: [batch_size, hidden_channels]
            return combined_embeddings

    # Initialize and move model to GPU
    max_len_src = max(
        max([len(values["source"]) for values in reac_train_sampled.values()]),
        max([len(values["destination"]) for values in reac_train_sampled.values()])
    )
    print(f"Max sequence length: {max_len_src}")

    model = EnhancedGCN(
        in_channels=data_.num_features,
        num_classes=len(unique_nodes),
        max_len_src=max_len_src
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # Prepare a mask tensor for edges
    num_edges = data_.edge_index.size(1)
    edge_mask = torch.ones(num_edges, dtype=torch.bool).to(device)  # All edges active initially

    # Modify the Data object to include the mask
    data_.edge_mask = edge_mask  # Adding edge mask to data

    # Build a mapping from edges to their indices
    edge_indices = {}
    for i in range(data_.edge_index.size(1)):
        src = int(data_.edge_index[0, i].item())
        tgt = int(data_.edge_index[1, i].item())
        edge_indices[(src, tgt)] = i

    # Training loop
    batch_size = 32  # Define your batch size
    num_epochs = 150  # Define the number of epochs

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        predicted_tokens = []

        # Prepare the data for batching
        reaction_items = list(reac_train_sampled.items())
        labels_list = labels  # Assuming labels is a list of missing nodes

        # Create batches
        for batch_start in range(0, len(reaction_items), batch_size):
            batch_end = min(batch_start + batch_size, len(reaction_items))
            batch_items = reaction_items[batch_start:batch_end]
            batch_labels = labels_list[batch_start:batch_end]

            # Collect missing nodes and reactions
            missing_nodes = []
            reaction_ids = []
            list1_batch = []
            list2_batch = []
            edges_to_remove = []

            for idx_in_batch, (reaction_id, values) in enumerate(batch_items):
                missing = batch_labels[idx_in_batch]
                reaction = reaction_id

                missing_nodes.append(missing)
                reaction_ids.append(reaction)

                # Prepare list1 and list2
                temp = values["source"].copy()
                if missing in temp:
                    temp.remove(missing)
                list1_batch.append(temp)
                list2_batch.append(list(values["destination"]))

                # Get edge to remove
                edge_to_remove = (missing, reaction)
                edge_to_remove_id = edge_indices.get(edge_to_remove)
                if edge_to_remove_id is not None:
                    edges_to_remove.append(edge_to_remove_id)

            # Deactivate edges
            data_.edge_mask[edges_to_remove] = False

            # Forward pass
            active_edges = data_.edge_index[:, data_.edge_mask]

            out = model(data_.x, active_edges, list1_batch=list1_batch, list2_batch=list2_batch)

            # Predicted tokens
            _, indices_found = torch.max(out, dim=1)  # indices_found: [batch_size]
            predicted_tokens.extend(indices_found.tolist())

            # Compute accuracy
            batch_labels_tensor = torch.tensor(missing_nodes, dtype=torch.long).to(device)
            correct += (indices_found == batch_labels_tensor).sum().item()

            # Compute loss and optimize
            optimizer.zero_grad()
            loss = criterion(out, batch_labels_tensor)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Reactivate edges
            data_.edge_mask[edges_to_remove] = True

        accuracy = correct / len(reac_train_sampled)
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")
        print(f"Predicted tokens: {predicted_tokens}")
        if (epoch % 10 == 0):
            model.eval()
            final_result,acc,hits_at_10,mmr_at_10 = validate_model2(model)
        return model

def create_pred_file(final_result : dict, file : str):
  data_read = read_the_graph(file)
  first_id = int(data_read["reaction id"][0])
  print(first_id)

  with open('Completion_test_answer.csv',"w") as file:
    i =0
    for _,value in final_result.items():
      reaction_id = first_id + i
      first_line : str = str(reaction_id) + ',{'
      

      for node in value :
        first_line += str(node) + ','
      
      first_line+='}\n'

      i+=1
      file.write(first_line)
  
  return 1

def evaluate_final_result(final_result, file_missing):
    missing = pd.read_csv(file_missing)["missing node"]
    missing = [int(n) for n in missing]


    
    acc= -1
    hits_at_10 = -1
    mmr_at_10 = -1


    number_of_correct_pred = 0
    number_of_true_pred = 0
    ranks = []  # To store ranks for MMR calculation
    i = 0
    for reaction_id, value in final_result.items():
          #print(reaction_id, ':', value)
          true_missing_node = missing[i]

          # Check if true missing node is in the top 10 predictions
          top_10_predictions = list(value.keys())
          if len(top_10_predictions) != 0 and top_10_predictions[0] == true_missing_node:
            number_of_true_pred +=1

          if true_missing_node in top_10_predictions:
              number_of_correct_pred += 1

              # Find the rank of the true missing node
              rank = top_10_predictions.index(true_missing_node) + 1  # 1-based index
              ranks.append(1 / rank)
          else:
              ranks.append(0)  # If not in top 10, reciprocal rank is 0
          i+=1

    hits_at_10 = number_of_correct_pred / len(final_result)
    mmr_at_10 = sum(ranks) / len(final_result)

    acc = number_of_true_pred / len(final_result)
  

    return final_result,acc,hits_at_10,mmr_at_10,

#Uses the best GNN model to compute the result file. Also computes the metrics on the vaildation dataset
#To use if you want to test the GNN model
def test_best_GNN():
  data =read_the_graph("dataset/Completion_training.csv")
  unique_nodes,map_nodes,map_node_reverse = get_unique_nodes(data)
  node_features,edge_index,labels,reac_train = get_final_node(data)
  DG,sources,all_nodes = graph_create(data)

  #Use the model
  #training and creation
  gat,new_edge_index,new_node_features = GAT_model_crea(DG,data,reac_train,unique_nodes,all_nodes)
  

  _,acc,hits_at_10,mmr_at_10 = validate_model2(all_nodes,new_node_features,new_edge_index,gat,validation_query='dataset/Completion_valid_query.csv',validation_answer='dataset/Completion_valid_answer.csv',batch_size=32)
  final_result,_,_,_ = validate_model2(all_nodes,new_node_features,new_edge_index,gat,validation_query='dataset/Completion_test_query.csv',validation_answer='',batch_size=32)
  #print(final_result)
  create_pred_file(final_result,'dataset/Completion_test_query.csv')



#Used to run the best Jacard model and create the result file
def test_best_model_Jacard():
    data =read_the_graph("dataset/Completion_training.csv")
    unique_nodes,map_nodes,map_node_reverse = get_unique_nodes(data)
    node_features,edge_index,labels,reac_train = get_final_node(data)
    DG,sources,all_nodes = graph_create(data)


    #Adar
    print('Adar------------------------------------------------------------------')
    _,acc,hits_at_10,mmr_at_10 = validate_adar(sources,DG,data,validation_query = 'dataset/Completion_valid_query.csv',validation_answer = 'dataset/Completion_valid_answer.csv')
    print(f"Validation Results - Accuracy: {acc:.4f}, Hits@10: {hits_at_10:.4f}, MMR@10: {mmr_at_10:.4f}")
    #Jacard
    print('Custom Jacard------------------------------------------------------------------')
    _,acc,hits_at_10,mmr_at_10,_= validate_custom(DG,data,reac_train,sources,validation_query = 'dataset/Completion_valid_query.csv',validation_answer = 'dataset/Completion_valid_answer.csv')
    print(f"Validation Results - Accuracy: {acc:.4f}, Hits@10: {hits_at_10:.4f}, MMR@10: {mmr_at_10:.4f}")
    #Create the test file

    final_result,_,_,_,_= validate_custom(DG,data,reac_train,sources,validation_query = 'dataset/Completion_test_query.csv',validation_answer='')
    
    create_pred_file(final_result,'dataset/Completion_test_query.csv')
    
    #print(final_result.keys())
    #print(acc,hits_at_10,mmr_at_10)






def main():
    ###Creating the graph ------------------------------------------------------
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Print the current working directory

    print("Current working directory:", os.getcwd())

    #Print testing the best GNN model for 150 epochs (5-6min)----------------------------------------------
    #test_best_GNN()
    test_best_model_Jacard()
    # data =read_the_graph("dataset/Completion_training.csv")
    # unique_nodes,map_nodes,map_node_reverse = get_unique_nodes(data)
    # node_features,edge_index,labels,reac_train = get_final_node(data)
    # DG,sources,all_nodes = graph_create(data)

    # #final_result,acc,hits_at_10,mmr_at_10 = validate_adar(sources,DG,data,validation_query = 'dataset/Completion_valid_query.csv',validation_answer = 'dataset/Completion_valid_answer.csv')
    # final_result,acc,hits_at_10,mmr_at_10,reactions= validate_custom(DG,data,reac_train,sources,validation_query = 'dataset/Completion_valid_query.csv',validation_answer = '')
    # #gat,new_edge_index,new_node_features = GAT_model_crea(DG,data,reac_train,unique_nodes,all_nodes)

    # #_,acc,hits_at_10,mmr_at_10 = validate_model2(all_nodes,new_node_features,new_edge_index,gat,validation_query='dataset/Completion_valid_query.csv',validation_answer='dataset/Completion_valid_answer.csv',batch_size=32)

    
    # create_pred_file(final_result,'dataset/Completion_valid_query.csv')
    # final_result,acc,hits_at_10,mmr_at_10 = evaluate_final_result(final_result,'dataset/Completion_valid_answer.csv')

    # print(final_result.keys())
    # print(acc,hits_at_10,mmr_at_10)


if __name__ == "__main__":
    main()
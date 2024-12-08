import pandas as pd
from collections import Counter
import ast
import pandas as pd
import ast
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx
from collections import defaultdict

# Fonction pour convertir les colonnes sources et destinations en ensembles de nombres
def parse_metabolites(column):
    return column.apply(lambda x: set(ast.literal_eval(x)) if pd.notnull(x) else set())

def compute_metrics(predictions, ground_truths):
    """
    Calcule le rappel, la précision et le F1-score pour une tâche multi-label.
    
    Args:
    - predictions : liste de listes des métabolites prédits (D_pred pour chaque réaction)
    - ground_truths : liste de listes des métabolites réels (D_real pour chaque réaction)

    Returns:
    - recall : rappel moyen
    - precision : précision moyenne
    - f1 : F1-score moyen
    """
    recalls, precisions, f1_scores = [], [], []

    for D_real, D_pred in zip(ground_truths, predictions):
        D_real = set(D_real)
        D_pred = set(D_pred)

        # Intersection entre métabolites réels et prédits
        intersection = len(D_real & D_pred)

        # Calcul des métriques pour cette réaction
        recall = intersection / len(D_real) if len(D_real) > 0 else 0
        precision = intersection / len(D_pred) if len(D_pred) > 0 else 0
        f1 = (
            2 * recall * precision / (recall + precision)
            if (recall + precision) > 0
            else 0
        )

        recalls.append(recall)
        precisions.append(precision)
        f1_scores.append(f1)

    # Moyennes globales
    mean_recall = sum(recalls) / len(recalls)
    mean_precision = sum(precisions) / len(precisions)
    mean_f1 = sum(f1_scores) / len(f1_scores)

    return mean_recall, mean_precision, mean_f1


def weight_union_networkx(validation_data, train_data, radius, response = None, eval_mode = False, increase_weight = 0.5, weight = 1) :
    # 1. Construction du graphe dirigé avec pondération
    G = nx.DiGraph()

    # Ajouter des arêtes avec des pondérations (par exemple, la fréquence des réactions)
    for _, row in train_data.iterrows():
        source_metabolites = row['source']
        destination_metabolites = row['destination']
        for source in source_metabolites:
            for destination in destination_metabolites:
                # Ajouter une arête pondérée en fonction de la fréquence (ou d'une autre métrique)
                if G.has_edge(source, destination):
                    G[source][destination]['weight'] += increase_weight
                else:
                    G.add_edge(source, destination, weight=weight)

    # 2. Recherche des destinations les plus probables (avec pondération)
    def get_predicted_destinations_with_weight(source, graph, radius=radius):
        reachable_nodes = set(nx.single_source_shortest_path_length(graph, source, cutoff=radius).keys())
        # Filtrer les destinations avec un seuil de pondération
        weighted_destinations = [node for node in reachable_nodes if graph[source].get(node, {}).get('weight', 0) > 1]
        return weighted_destinations

    # 3. Prédiction des destinations
    predicted_destinations = []
    for _, row in validation_data.iterrows():
        all_predicted = set()
        for source_metabolite in row['source']:
            destinations = get_predicted_destinations_with_weight(source_metabolite, G)
            all_predicted.update(destinations)

        # Si aucune destination n'est trouvée, ajouter 0
        if not all_predicted:
            all_predicted.add(0)

        predicted_destinations.append(all_predicted)

    validation_data['predicted_destination'] = predicted_destinations

    if eval_mode : 
        # Exemple pour évaluer une réaction donnée (true et predicted)
        true_destinations = response['destination'].tolist()  # Remplacez par vos vraies valeurs
        predicted_destinations = validation_data['predicted_destination'].tolist()  # Remplacez par vos prédictions

        validation_data['real_destination'] = true_destinations
        validation_data["contains_answer"] = validation_data.apply(lambda row : len(row["predicted_destination"].intersection(row["real_destination"])) / len(row["real_destination"]), axis = 1)
        validation_data["size_pred"] = validation_data.apply(lambda row : len(row["predicted_destination"]), axis = 1)

        mean_recall, mean_precision, mean_f1 = compute_metrics(predicted_destinations, true_destinations)
        return radius, weight, increase_weight, mean_recall, mean_precision, mean_f1, validation_data["contains_answer"].mean(), validation_data["size_pred"].mean()


def raw_data_to_binary(dataset) :

    binary_data = []

    for _, row in dataset.iterrows(): # for each line
        source_set = row["source"]
        destination_set = row["destination"]

        for destination_node in destination_set: # positive data
            binary_data.append({"source": source_set, "node": destination_node, "label": 1})

        negative_candidates = all_nodes - set(source_set) - set(destination_set) # negative data
        num_negatives = round(len(destination_set) * 2.5)  # Équilibrage
        negative_samples = random.sample(list(negative_candidates), min(num_negatives, len(negative_candidates)))

        for negative_node in negative_samples:
            binary_data.append({"source": source_set, "node": negative_node, "label": 0})

    return pd.DataFrame(binary_data)

def collate_fn(batch): # sert à uniformiser la longueur des dimensions dans le batch
    # Trouver la longueur maximale des sources dans le batch
    max_len = max([len(source) for source, _, _ in batch])
    
    padded_sources = []
    nodes = []
    labels = []
    
    for source, node, label in batch:
        # Ajouter le padding aux sources
        padding = torch.zeros(max_len - len(source), dtype=torch.long)
        padded_sources.append(torch.cat([source, padding]))
        nodes.append(node)
        labels.append(label)
    
    # Empiler les tensors
    sources_tensor = torch.stack(padded_sources)
    nodes_tensor = torch.stack(nodes)
    labels_tensor = torch.stack(labels)
    
    return sources_tensor, nodes_tensor, labels_tensor

def evaluate_model(model, dataloader):
    model.eval()
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for source, node, label in dataloader:
            probs = model(source, node)
            all_labels.extend(label.numpy())
            all_probs.extend(probs.numpy())
    
    # Utilisez les métriques comme précision, rappel et AUC
    from sklearn.metrics import precision_score, recall_score, roc_auc_score
    threshold = 0.5
    predictions = [1 if p > threshold else 0 for p in all_probs]
    precision = precision_score(all_labels, predictions)
    recall = recall_score(all_labels, predictions)
    auc = roc_auc_score(all_labels, all_probs)
    return precision, recall, auc




class BinaryReactionDataset(Dataset):
    def __init__(self, data, num_nodes):
        """
        data: DataFrame contenant 'source', 'node', 'label'
        num_nodes: Nombre total de nodes dans le graphe
        """
        self.data = data
        self.num_nodes = num_nodes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        source = row['source']  # Liste des nodes dans la source
        node = row['node']      # Un node candidat (on étudie si il y a un lien entre ce node spécifique et la source)
        label = row['label']    # 0 ou 1 (le label à déterminer)
        
        # Convertir source en tenseur (liste d'indices)
        source_tensor = torch.tensor(source, dtype=torch.long)
        node_tensor = torch.tensor(node, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.float)
        
        return source_tensor, node_tensor, label_tensor
    
    
    
class ReactionEmbeddingModel(nn.Module):
    def __init__(self, num_nodes, embedding_dim, hidden_dim, output_dim):
        super(ReactionEmbeddingModel, self).__init__()
        self.node_embedding = nn.Embedding(num_nodes, embedding_dim)
        self.fc1 = nn.Linear(2 * embedding_dim, hidden_dim)  # Changement ici
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, source, node):
        source_emb = self.node_embedding(source)
        node_emb = self.node_embedding(node)
        source_emb = source_emb.mean(dim=1)
        
        # Concatenation au lieu du produit élément par élément
        combined = torch.cat([source_emb, node_emb], dim=1)  # [batch_size, 2 * embedding_dim]

        x = self.fc1(combined).relu()
        x = self.bn1(x)
        x = self.fc2(x).relu()
        x = self.bn2(x)
        x = self.fc3(x).relu()
        x = self.dropout(x)
        x = self.fc4(x).squeeze()
        return torch.sigmoid(x)
    
def link_prediction(validation_data, response = None, eval_mode = False, p_threshold = 0.5) :
    binary_data = []

    for i, row in validation_data.iterrows(): # for each line
        
        source_set = row["source"]
        destination_set = row["predicted_destination"]
        react_id = row["reaction id"] # pour la reconstruction du df avec les set() prédis

        for destination_node in destination_set : # positive data
            
            if eval_mode :
                
                truth = response["destination"].iloc[i]
                
                if destination_node in truth :
                    binary_data.append({"react_id": react_id, "source": list(source_set), "node": destination_node, "label": 1})
                else :
                    binary_data.append({"react_id": react_id, "source": list(source_set), "node": destination_node, "label": 0})

            else :
                binary_data.append({"react_id": react_id, "source": list(source_set), "node": destination_node,"label" : 0})

    val_binary_df = pd.DataFrame(binary_data)
    val_dataset = BinaryReactionDataset(val_binary_df, num_nodes)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn) # dataloader of batch = 32

    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for source, node, label in val_dataloader:
            probs = model(source, node)
            all_labels.extend(label.numpy())
            all_probs.extend(probs.numpy())

    val_binary_df["prob"] = all_probs

    # Convertir 'source' en tuple pour qu'il soit hashable
    val_binary_df["source"] = val_binary_df["source"].apply(tuple)

    grouped = val_binary_df.groupby('react_id').agg({
        'node': list,  # Collecte les nodes dans une liste
        'prob': list  # Collecte les prédictions dans une liste (facultatif)
    }).reset_index()

    # Appliquer un masque pour garder uniquement les nodes où predictions == 1
    grouped['filtered_nodes'] = grouped.apply(
        lambda row: {n for n, p in zip(row['node'], row['prob']) if p >= p_threshold},
        axis=1
    )
    grouped['filtered_nodes'] = grouped['filtered_nodes'].apply(lambda row : {0} if len(row) == 0 else row)
    
    test['destination'] = grouped['filtered_nodes']

    if eval_mode :
        true_destinations = response['destination'].tolist()
        grouped['real_destination'] = true_destinations
        grouped["contains_answer"] = grouped.apply(lambda row : len(row["filtered_nodes"].intersection(row["real_destination"])) / len(row["real_destination"]), axis = 1)
        grouped["size_pred"] = grouped.apply(lambda row : len(row["filtered_nodes"]), axis = 1)

        predicted_destinations = grouped['filtered_nodes'].tolist() 
        true_destinations = grouped['real_destination'].tolist() 
    
        mean_recall, mean_precision, mean_f1 = compute_metrics(predicted_destinations, true_destinations)

        return p_threshold, mean_recall, mean_precision, mean_f1, grouped["contains_answer"].mean(), grouped["size_pred"].mean()

    
    
###################################################################
    
    
# Charger les données
train_file = './dataset/Prediction_training.csv'  # Remplacez par le chemin de votre fichier d'entraînement
validation_file = './dataset/Prediction_valid_query.csv'  # Remplacez par le chemin de votre fichier de validation
response_file = './dataset/Prediction_valid_answer.csv'
test_file = './dataset/Prediction_test_query.csv'

# Chargement des fichiers CSV
train_data = pd.read_csv(train_file)
validation_data = pd.read_csv(validation_file)
response = pd.read_csv(response_file)
test = pd.read_csv(test_file)

# Conversion des colonnes source et destination
train_data['source'] = parse_metabolites(train_data['source'])
train_data['destination'] = parse_metabolites(train_data['destination'])
validation_data['source'] = parse_metabolites(validation_data['source'])
response['destination'] = parse_metabolites(response['destination'])
test['source'] = parse_metabolites(test['source'])

# Initialisation
source_to_destination = defaultdict(set)

# Construire le dictionnaire de co-occurrences
for _, row in train_data.iterrows():
    for source_metabolite in row['source']:
        source_to_destination[source_metabolite].update(row['destination'])

# Prédiction des destinations pour les données de validation
predicted_destinations = []

for _, row in validation_data.iterrows():
    candidates = set()
    for source_metabolite in row['source']:
        if source_metabolite in source_to_destination:
            candidates.update(source_to_destination[source_metabolite])
    # Si aucun candidat n'est trouvé, ajouter la destination 0
    if not candidates:
        candidates.add(0)
    predicted_destinations.append(candidates)

validation_data['predicted_destination'] = predicted_destinations

# Exemple pour évaluer une réaction donnée (true et predicted)
true_destinations = response['destination'].tolist()  # Remplacez par vos vraies valeurs
predicted_destinations = validation_data['predicted_destination'].tolist()  # Remplacez par vos prédictions


validation_data['real_destination'] = true_destinations
validation_data["contains_answer"] = validation_data.apply(lambda row : len(row["predicted_destination"].intersection(row["real_destination"])) / len(row["real_destination"]), axis = 1)
validation_data["size_pred"] = validation_data.apply(lambda row : len(row["predicted_destination"]), axis = 1)

mean_recall, mean_precision, mean_f1 = compute_metrics(predicted_destinations, true_destinations)
print("====PRESELECTION====")
print("Recall", mean_recall)
print("Precision", mean_precision)
print("F1-Score", mean_f1)

validation_data["contains_answer"].mean(), validation_data["size_pred"].mean()

# DATA CONSTRUCTION 

train_data["source"] = train_data["source"].apply(lambda x: list(x)) # bonne lecture des données
train_data["destination"] = train_data["destination"].apply(lambda x: list(x))

all_nodes = set() # liste de tous les nodes possibles
train_data["source"].apply(lambda x: all_nodes.update(x))
train_data["destination"].apply(lambda x: all_nodes.update(x))

binary_df = raw_data_to_binary(train_data) # construct the dataframe

num_nodes = len(all_nodes) # nombre total de nodes (cf : data construction)

dataset = BinaryReactionDataset(binary_df, num_nodes)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn) # dataloader of batch = 32


# MODEL
embedding_dim = 200 # 128
hidden_dim = 256  # Increased hidden layer size
output_dim = 1
learning_rate = 0.0001
num_epochs = 15

# Initialisation du modèle, de l'optimiseur et de la fonction de perte
model = ReactionEmbeddingModel(num_nodes=num_nodes, embedding_dim=embedding_dim, hidden_dim=hidden_dim, output_dim=output_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.BCELoss()

# Boucle d'entraînement
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for source, node, label in dataloader :
        
        optimizer.zero_grad()
        # Calculer les prédictions
        probs = model(source, node)
        # Calculer la perte
        loss = loss_fn(probs, label)
        # Backpropagation
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")


# on trained data
precision, recall, auc = evaluate_model(model, dataloader) #métrique sur les données entrainées
print("====LINK PREDICTION====")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, AUC: {auc:.4f}")

    
print("====VALIDATION====")
weight_union_networkx(validation_data, train_data, radius = 1, response = response, eval_mode = True)
_, mean_recall, mean_precision, mean_f1, _, _ = link_prediction(validation_data, response, eval_mode = True, p_threshold = 0.99)
print(f"Precision: {mean_precision:.4f}, Recall: {mean_recall:.4f}, F1-score: {mean_f1:.4f}")

print("====TEST====")
weight_union_networkx(test, train_data, radius = 1, eval_mode = False)
link_prediction(test, p_threshold = 0.99)

output = test[['reaction id', 'destination']]
print(output.head(5))

output.to_csv('./dataset/Prediction_test_answer.csv', index=False)

print("Output saved : ./dataset/Prediction_test_answer.csv")







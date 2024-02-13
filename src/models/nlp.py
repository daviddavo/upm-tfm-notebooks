from typing import Optional, Union

from pathlib import Path

import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer

from .. import paths

def get_embeddings_from_cache(dfp, model, embeddings_cache=None):
    if embeddings_cache == None:
        embeddings_cache = paths.pln_embeddings_cache(model)

    if embeddings_cache.exists():
        prev_embeddings = pd.read_pickle(embeddings_cache)
        remaining_embeddings_idx = dfp.index.difference(prev_embeddings.index)
    else:
        prev_embeddings = pd.Series(dtype=str)
        remaining_embeddings_idx = dfp.index

    if not remaining_embeddings_idx.empty:
        print("Some embeddings need to be calculated")
        remaining = dfp.loc[remaining_embeddings_idx]
        title_description = remaining['title'] + '\n' + remaining['description']

        new_embeddings = pd.Series(
            list(model.encode(title_description, show_progress_bar=True, normalize_embeddings=True)),
            index=title_description.index,
        )

        all_embeddings = pd.concat((prev_embeddings, new_embeddings))
        all_embeddings.to_pickle(embeddings_cache)
    else:
        print("All embeddings are already calculated")
        all_embeddings = prev_embeddings
    
    return all_embeddings.loc[dfp.index]

class NLPModel:
    def __init__(self, 
        dfp: pd.DataFrame,
        *,
        transformer_model: str ='all-mpnet-base-v2',
        show_progress_bar: bool = True,
        embeddings_cache: Optional[Union[str, Path]] = None,
    ):
        self.dfp = dfp
        self.transformer_model = SentenceTransformer(transformer_model)
        self.show_progress_bar = show_progress_bar
        self.embeddings_cache: Path = embeddings_cache or paths.pln_embeddings_cache(transformer_model)
        self.embeddings = None

    def fit(self):
        self.embeddings = get_embeddings_from_cache(self.dfp, self.transformer_model, self.embeddings_cache)

class NLPSimilarity(NLPModel):
    def __init__(self,
        train: pd.DataFrame,
        dfp: pd.DataFrame,
        voter_col: str = 'userID',
        proposal_col: str = 'itemID',
        **kwargs,
    ):
        super().__init__(dfp, **kwargs)
        self.train = train.reset_index(drop=True)
        self.voter_col = voter_col
        self.proposal_col = proposal_col

    def fit(self):
        super().fit()
        votes_embeddings = self.embeddings.loc[self.train[self.proposal_col]]
        self.voter_embeddings = self.train.groupby(self.voter_col).apply(lambda x: votes_embeddings[x.index].sum(axis=0))

    def recommend_k_items(
        self, to_users, top_k=5, remove_seen=True, recommend_from=None, min_score = 0.0,
    ):
        voter_embeddings = self.voter_embeddings.loc[to_users]
        np_voter_embeddings = np.stack(voter_embeddings.to_numpy())
    
        prop_embeddings = self.embeddings
        if recommend_from is not None:
            assert len(recommend_from) >= top_k, "top_k should not be greater than the number of proposals to recommend"
            prop_embeddings = self.embeddings.loc[recommend_from]

        tr_embeddings = np.stack(prop_embeddings.to_numpy())
    
        scores = np_voter_embeddings @ tr_embeddings.T

        if remove_seen:
            trainu = self.train[self.train['userID'].isin(voter_embeddings.index) & self.train['itemID'].isin(prop_embeddings.index)]
            itemID2idx = pd.Series(data=np.arange(len(prop_embeddings)), index=prop_embeddings.index)
            voterID2idx = pd.Series(data=np.arange(len(voter_embeddings)), index=voter_embeddings.index)

            scores[voterID2idx[trainu['userID']], itemID2idx[trainu['itemID']]] = -np.inf
            
        best = (-scores).argsort(axis=1)
        topk = best[:, :top_k]

        # create df with columns
        # userID, itemID, prediction
        uid = np.repeat(np.arange(np_voter_embeddings.shape[0]), top_k)
        iid = topk.flatten()

        # transform int to id
        df = pd.DataFrame({
            'userID': voter_embeddings.index[uid],
            'itemID': prop_embeddings.index[iid].astype(str),
            # 'prediction': 1,
            'prediction': scores[uid, iid],
        })
        return df[df['prediction'] > min_score].reset_index(drop=True)

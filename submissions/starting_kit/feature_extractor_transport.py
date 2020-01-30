import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline



class FeatureExtractor(object):
	def __init__(self):
		pass

	def fit(self, X_df, y_array):
		allfiles = os.listdir('./data/data valid/')
		li = []

		for file in allfiles :
			#print(file)
			file = './data/data valid/' + file
			if file[-3:] == 'csv':
				data = pd.read_csv(file, header=0, sep=';')
				data['JOUR'] = pd.to_datetime(data['JOUR'])
				li.append(data)
			else:
				data = pd.read_csv(file, header=0, sep='\t')
				data['JOUR'] = pd.to_datetime(data['JOUR'])
				li.append(data)

		dt = pd.concat(li, axis=0, ignore_index=True)
		# obtain features from award

		def nb_vald(X):
			X.loc[X['NB_VALD'] == 'Moins de 5', 'NB_VALD'] = 5
			X['NB_VALD'] = pd.to_numeric(X['NB_VALD'])

			return X
		nb_vald_transformer = FunctionTransformer(nb_vald, validate=False)



		def process_jour(X):
			X['JOUR'] = X['JOUR'].astype(str)
			X = pd.DataFrame(X.groupby(["JOUR","LIBELLE_ARRET"])['NB_VALD'].sum().reset_index())
			return X
		jour_transformer = FunctionTransformer(process_jour, validate=False)



		def merge_pos(X):
			pos = pd.read_csv("../positions-geographiques-des-stations-du-reseau-ratp.csv", sep=';')
			dt2 = pos[['Name','Coordinates']].copy()

			l = np.array(dt2["Coordinates"].str.split(",").tolist())
			dt2["lon"] = l[:,0]
			dt2["lat"] = l[:,1]
			dt2.drop("Coordinates",axis=1, inplace=True)

			dt2['Name'] = dt2['Name'].str.lower()
			dt2['Name'] = dt2['Name'].str.replace('[^\w]','')
			dt2['Name'] = dt2['Name'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

			dt2['lon'] = pd.to_numeric(dt2["lon"])
			dt2['lat'] = pd.to_numeric(dt2["lat"])

			dt2 = dt2.groupby('Name').mean().reset_index()
			dt2.drop_duplicates(inplace=True)

			X['Name'] = X['LIBELLE_ARRET'].str.lower()
			X['Name'] = X['Name'].str.replace('[^\w]','')
			X['Name'] = X['Name'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

			dt_agg_pos = X.merge(dt2, how='left', on='Name')
			dt_agg_pos.drop('Name',axis=1,inplace=True)


			return dt_agg_pos
		merge_pos_transformer = FunctionTransformer(merge_pos, validate=False)

		def process_date(X):
			date = pd.to_datetime(X['JOUR'], format='%Y-%m-%d')
			return np.c_[date.dt.year, date.dt.month, date.dt.day]
		date_transformer = FunctionTransformer(process_date, validate=False)

		def merge_corr(X):
			corres = pd.read_csv('data/correspondances.csv')
			corr = corres.groupby('stop_name').agg({'rer' : sum,'line' : pd.Series.nunique}).reset_index().rename(columns={'rer':'nb_rer','line':'nb_metro'})
			corr['nb_metro'] = corr['nb_metro'] - corr['nb_rer']

			corr['Name'] = corr['stop_name'].str.lower()
			corr['Name'] = corr['Name'].str.replace('[^\w]','')
			corr['Name'] = corr['Name'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

			X['Name'] = X['LIBELLE_ARRET'].str.lower()
			X['Name'] = X['Name'].str.replace('[^\w]','')
			X['Name'] = X['Name'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

			dt_agg_pos_corr = X.merge(corr, how='left', on='Name')
			dt_agg_pos_corr.drop(['Name','stop_name'],axis=1,inplace=True)


			return dt_agg_pos_corr
		merge_corr_transformer = FunctionTransformer(merge_corr, validate=False)


		def merge_incident(X):

			inc = pd.read_csv('./data/incidents_2016_2019_clean.csv')
			inc['day'] = inc['date'].str[:10]
			inc['duree'] = pd.to_timedelta(inc['duree'])

			inc_group = inc.groupby(['day','line_num']).agg({'duree' : sum,'what' : 'first','type_inc' : 'count'}).reset_index().rename(columns={'type_inc':'num_inc','what':'type_inc'})
			inc_group['day'] = pd.to_datetime(inc_group['day'])
			inc_station = inc_group.merge(corres, how='left',left_on='line_num',right_on='line').drop(['line','rer'],axis=1)

			inc_station['Name'] = inc_station['stop_name'].str.lower()
			inc_station['Name'] = inc_station['Name'].str.replace('[^\w]','')
			inc_station['Name'] = inc_station['Name'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

			inc_station['day'] = inc_station['day'].astype(str)

			curr_dt = X.copy()

			curr_dt['Name'] = curr_dt['LIBELLE_ARRET'].str.lower()
			curr_dt['Name'] = curr_dt['Name'].str.replace('[^\w]','')
			curr_dt['Name'] = curr_dt['Name'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

			curr_dt['JOUR'] = curr_dt['JOUR'].astype(str)
			dt_pos_cor_inc = curr_dt.merge(inc_station, how='left',left_on=['JOUR','Name'], right_on=['day','Name'])
			dt_pos_cor_inc.drop(['Name','day','line_num','stop_name'],axis=1,inplace=True)
			dt_pos_cor_inc.loc[dt_pos_cor_inc['num_inc'].notna()]

			return dt_pos_cor_inc
		merge_incident_transformer = FunctionTransformer(merge_incident, validate=False)

		def merge_jour_ferier(X):

			data_ferie = pd.read_csv("./data/jours-feries.csv")
			# Let's convert dates into datetime type
			data_ferie['date'] = pd.to_datetime(data_ferie['date'],format="%Y/%m/%d")
			# Let's keep only dates between 2015-01-01 and 2019-06-30
			data_ferie=data_ferie[data_ferie['date'].isin(pd.date_range(start='20150101', end='20190630'))]
			# Let's replace NaN values by "jour non ferie"
			data_ferie["nom_jour_ferie"].fillna("Jour non ferie", inplace = True)

			data_ferie.columns = ['date','est_jour_ferie','type_jour']
			data_ferie.est_jour_ferie = data_ferie.est_jour_ferie.astype(int)
			data_ferie['is_weekend'] = data_ferie['date'].map(lambda x : 1 if x.weekday() >= 5 else 0 )

			#Ajouter le type "Weekend" pour la colonne "type_jour"
			mask = (data_ferie['is_weekend'] == 1)
			data_ferie['type_jour'][mask] = "Weekend"
			data_ferie.drop('is_weekend',axis=1,inplace=True)

			curr_dt = X.copy()
			curr_dt['JOUR'] = pd.to_datetime(curr_dt['JOUR'])
			df_merged = pd.merge(curr_dt, data_ferie, left_on='JOUR',right_on='date',how='left')
			df_merged.drop('date',axis=1,inplace=True)



			return df_merged
		merge_ferier_transformer = FunctionTransformer(merge_jour_ferier, validate=False)

		def merge_ref(X):
			ref = pd.read_csv('data/referentiel-gares-voyageurs.csv',delimiter = ';')
			ref  = ref[['Intitulé gare','Segment DRG','Nbre plateformes']]
			ref['Intitulé gare'] = ref['Intitulé gare'].apply(lambda x : x.upper())

			dt_ref = pd.merge(X,ref,left_on='LIBELLE_ARRET',right_on='Intitulé gare',how = 'left')
			dt_ref = dt_ref.drop(columns = ['Intitulé gare'],axis=1)
			return df_ref
		merge_ref_transformer = FunctionTransformer(merge_ref, validate=False)

		def merge_mvt_so(X):
			mvs = pd.read_csv('data/mouvements-sociaux.csv',delimiter=';')
			mvs['date_de_debut'] = pd.to_datetime(mvs['date_de_debut'],format="%Y/%m/%d")
			mvs['date_de_fin'] = pd.to_datetime(mvs['date_de_fin'],format="%Y/%m/%d")
			mvs  = mvs[mvs['date_de_debut'].isin(pd.date_range(start='20150101', end='20190630'))]
			mvs_dates = mvs[['date_de_debut','date_de_fin']]

			list_dates = []
			for index in mvs_dates.index:
				start_date = mvs_dates['date_de_debut'][index]
				if  not pd.isna(mvs_dates['date_de_fin'][index]):
					end_date = mvs_dates['date_de_fin'][index]
					for n in range(int ((end_date - start_date).days)):
						list_dates.append(start_date + timedelta(n))
				else :
					list_dates.append(start_date)

			mvs_dates = {'date': list_dates}
			mvs_dates = pd.DataFrame(data=mvs_dates)
			mvs_dates['est_greve'] = np.ones(mvs_dates.shape[0])

			X['JOUR'] = pd.to_datetime(X['JOUR'])
			df_merged = pd.merge(X,mvs_dates, left_on='JOUR',right_on='date',how='left')
			df_merged = df_merged.drop(columns=['date'])
			df_merged["est_greve"].fillna(0, inplace = True)
			return df_merged
		merge_mvt_so_transformer = FunctionTransformer(merge_mvt_so, validate=False)


		num_cols = ['NB_VALD','lon','lat','nb_rer','nb_metro','duree','num_inc','est_jour_ferie']
		nb_vald_col = ['NB_VALD']
		jour_col = ['JOUR']

		merge_pos_col = ['Name']
		merge_corr_col = ['Name']
		merge_incident_col = ['JOUR','Name']
		merge_jour_ferier_col = ['JOUR']
		merge_ref_col = ['LIBELLE_ARRET']
		merge_mvt_so_col = ['JOUR']


		preprocessor = ColumnTransformer(
			transformers=[
				('NB_VALD', make_pipeline(nb_vald_transformer,
				 SimpleImputer(strategy='median')), nb_vald_col),
				('JOUR',  make_pipeline(jour_transformer,
				 SimpleImputer(strategy='median')), jour_col),
				#('date', make_pipeline(date_transformer,
				# SimpleImputer(strategy='median')), date_cols),
				('merge_pos', make_pipeline(merge_pos_transformer,
				 SimpleImputer(strategy='median')), merge_pos_col),
				('merge_corr', make_pipeline(merge_corr_transformer,
				 SimpleImputer(strategy='median')), merge_corr_col),
				('merge_incident', make_pipeline(merge_incident_transformer,
				 SimpleImputer(strategy='median')), merge_incident_col),
				('merge_jour_ferier', make_pipeline(merge_ferier_transformer,
				 SimpleImputer(strategy='median')), merge_jour_ferier_col),
				('merge_ref', make_pipeline(merge_ref_transformer,
				 SimpleImputer(strategy='median')), merge_ref_col),
				('merge_mvt_so', make_pipeline(merge_mvt_so_transformer,
				 SimpleImputer(strategy='median')), merge_mvt_so_col),

				])

		self.preprocessor = preprocessor
		self.preprocessor.fit(X_df, y_array)
		return self

	def transform(self, X_df):
		return self.preprocessor.transform(X_df)

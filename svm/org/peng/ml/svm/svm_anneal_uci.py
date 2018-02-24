from sklearn.preprocessing import LabelBinarizer;
from sklearn.preprocessing import LabelEncoder;
import pandas as pd;
from sklearn.svm import SVC;
import numpy as np;
from sklearn.model_selection import cross_val_score;


anneal_file = 'c:/workspace/MachineLearningPlayground/data/anneal/anneal.data';
header = ['family', 'product-type', 'steel','carbon',
              'hardness', 'temper_rolling', 'condition',
              'formability', 'strength', 'non-ageing','surface-finish',
              'surface-quality', 'enamelability', 'bc','bf','bt','bw/me',
              'bl','m','chrom','phos', 'cbond', 'marvi', 'exptl','ferro',
              'corr', 'blue/bright/varn/clean', 'lustre', 'jurofm',
              's', 'p', 'shape', 'thick', 'width', 'len', 'oil', 'bore',
              'packing','classes']
data_header = ['family_code','product-type_code','steel_code','temper_rolling_code',
               'condition_code','formability_code','non-ageing_code','surface-finish_code',
               'surface-quality_code','enamelability_code','bc_code','bf_code',
               'bt_code','bw/me_code','bl_code','m_code','chrom_code','phos_code',
               'cbond_code','marvi_code','exptl_code','ferro_code','corr_code',
               'blue/bright/varn/clean_code','lustre_code','jurofm_code','s_code',
               'p_code','shape_code','oil_code','packing_code'];
label_header = 'classes_code';

def anneal_preprocessing():


    classes = ['1','2','3','4','5','U'];
    #Replace missing data
    df = pd.read_csv(anneal_file,names = header, index_col = False);
    df.loc[:,'family'] = df['family'].replace(to_replace="?", value= '');
    df.loc[:,'product-type'] = df['product-type'].replace(to_replace="?", value='');
    df.loc[:,'steel'] = df['steel'].replace(to_replace='?',value='');
    df.loc[:,'temper_rolling'] = df['temper_rolling'].replace(to_replace='?',value='');
    df.loc[:,'condition'] = df['condition'].replace(to_replace='?',value='');
    df.loc[:,'formability'] = df['formability'].replace(to_replace='?',value='');
    df.loc[:, 'non-ageing'] = df['non-ageing'].replace(to_replace='?', value='');
    df.loc[:, 'surface-quality'] = df['surface-quality'].replace(to_replace='?', value='');
    df.loc[:, 'surface-finish'] = df['surface-finish'].replace(to_replace='?', value='');
    df.loc[:, 'enamelability'] = df['enamelability'].replace(to_replace='?', value='');
    df.loc[:, 'bc'] = df['bc'].replace(to_replace='?', value='');
    df.loc[:, 'bf'] = df['bf'].replace(to_replace='?', value='');
    df.loc[:, 'bt'] = df['bt'].replace(to_replace='?', value='');
    df.loc[:, 'bw/me'] = df['bw/me'].replace(to_replace='?', value='');
    df.loc[:, 'bl'] = df['bl'].replace(to_replace='?', value='');
    df.loc[:, 'm'] = df['m'].replace(to_replace='?', value='');
    df.loc[:, 'chrom'] = df['chrom'].replace(to_replace='?', value='');
    df.loc[:, 'phos'] = df['phos'].replace(to_replace='?', value='');
    df.loc[:, 'cbond'] = df['cbond'].replace(to_replace='?', value='');
    df.loc[:, 'marvi'] = df['marvi'].replace(to_replace='?', value='');
    df.loc[:, 'exptl'] = df['exptl'].replace(to_replace='?', value='');
    df.loc[:, 'ferro'] = df['ferro'].replace(to_replace='?', value='');
    df.loc[:, 'corr'] = df['corr'].replace(to_replace='?', value='');
    df.loc[:, 'blue/bright/varn/clean'] = df['blue/bright/varn/clean'].replace(to_replace='?', value='');
    df.loc[:, 'lustre'] = df['lustre'].replace(to_replace='?', value='');
    df.loc[:, 'jurofm'] = df['jurofm'].replace(to_replace='?', value='');
    df.loc[:, 's'] = df['s'].replace(to_replace='?', value='');
    df.loc[:, 'p'] = df['p'].replace(to_replace='?', value='');
    df.loc[:, 'shape'] = df['shape'].replace(to_replace='?', value='');
    df.loc[:, 'oil'] = df['oil'].replace(to_replace='?', value='');
    df.loc[:, 'packing'] = df['packing'].replace(to_replace='?', value='');

    #Labelencoder
    lbc = LabelEncoder();
    lbb = LabelBinarizer();
    df['family_code'] = lbc.fit_transform(df['family']);
    df['product-type_code'] = lbc.fit_transform(df['product-type']);
    df['steel_code'] = lbc.fit_transform(df['steel']);
    df['temper_rolling_code'] = lbc.fit_transform(df['temper_rolling']);
    df['condition_code'] = lbc.fit_transform(df['condition']);
    df['formability_code'] = lbc.fit_transform(df['formability']);
    df['non-ageing_code'] = lbc.fit_transform(df['non-ageing']);
    df['surface-finish_code'] = lbc.fit_transform(df['surface-finish']);
    df['surface-quality_code'] = lbc.fit_transform(df['surface-quality']);
    df['enamelability_code'] = lbc.fit_transform(df['enamelability']);
    df['bc_code'] = lbc.fit_transform(df['bc']);
    df['bf_code'] = lbc.fit_transform(df['bf']);
    df['bt_code'] = lbc.fit_transform(df['bt']);
    df['bw/me_code'] = lbc.fit_transform(df['bw/me']);
    df['bl_code'] = lbc.fit_transform(df['bl']);
    df['m_code'] = lbc.fit_transform(df['m']);
    df['chrom_code'] = lbc.fit_transform(df['chrom']);
    df['phos_code'] = lbc.fit_transform(df['phos']);
    df['cbond_code'] = lbc.fit_transform(df['cbond']);
    df['marvi_code'] = lbc.fit_transform(df['marvi']);
    df['exptl_code'] = lbc.fit_transform(df['exptl']);
    df['ferro_code'] = lbc.fit_transform(df['ferro']);
    df['corr_code'] = lbc.fit_transform(df['temper_rolling']);
    df['blue/bright/varn/clean_code'] = lbc.fit_transform(df['blue/bright/varn/clean']);
    df['lustre_code'] = lbc.fit_transform(df['lustre']);
    df['jurofm_code'] = lbc.fit_transform(df['jurofm']);
    df['s_code'] = lbc.fit_transform(df['s']);
    df['p_code'] = lbc.fit_transform(df['p']);
    df['shape_code'] = lbc.fit_transform(df['shape']);
    df['oil_code'] = lbc.fit_transform(df['oil']);
    df['packing_code'] = lbc.fit_transform(df['packing']);
    df['classes_code'] = lbc.fit_transform(df['classes']);


    return df;


def anneal_svm(df):
    clf = SVC();
    clf.fit(X = df[data_header],y = df[label_header]);
    scores = cross_val_score(clf, df[data_header],df[label_header],cv = 10);
    print(scores);
    print("Accuracy: %0.2f (+/- %0.2f)", (scores.mean(), scores.std()*2));

df = anneal_preprocessing();
anneal_svm(df);
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <xgboost/c_api.h>

#define safe_xgboost(call) { \
  int err = (call); \
  if (err != 0) { \
    fprintf(stderr, "%s:%d: error in %s: %s\n", __FILE__, __LINE__, #call, XGBGetLastError()); \
    exit(1); \
  } \
}

/* Define a function to get CF, HF boosters */
void xgb_get_models(const char *cf_model_path, const char *hf_model_path,
		    BoosterHandle *cf_booster, BoosterHandle *hf_booster) {
  /* Create CF, HF boosters at the designated handles */
  safe_xgboost(XGBoosterCreate(NULL, 0, cf_booster));
  safe_xgboost(XGBoosterCreate(NULL, 0, hf_booster));

  /* Read in the CF, HF models */
  safe_xgboost(XGBoosterLoadModel(*cf_booster, cf_model_path));
  safe_xgboost(XGBoosterLoadModel(*hf_booster, hf_model_path));
}

const float *log_cf, *log_hf;

/* Define a function to get cfun, hfun @ Z=0 */
void xgb_get_chf(const float *Tem, const float *Hden, const float *Plw, const float *Ph1,
		 const float *Pg1, const float *Pc6, BoosterHandle *cf_booster, BoosterHandle *hf_booster,
		 float *cfun, float *hfun, int *err) {
  /* Create an XGBoost DMatrix object to hold the set of features for the XGBoost models */
  DMatrixHandle feat_dmatrix;

  /* Calculate and scale features */
  float t_feat = (log10(*Tem) - (1.000000)) / (9.000000 - (1.000000));
  float n_h_feat = (log10(*Hden) - (-6.000000)) / (6.000000 - (-6.000000));
  float q_lw_feat = (log10(*Plw / *Hden) - (-14.940437)) / (-2.822464 - (-14.940437));
  float q_hi_feat = (log10(*Ph1 / *Plw) - (-6.911031)) / (0.481141 - (-6.911031));
  float q_hei_feat = (log10(*Pg1 / *Plw) - (-5.551412)) / (0.717863 - (-5.551412));
  float q_cvi_feat = (log10(*Pc6 / *Plw) - (-9.017760)) / (-1.062396 - (-9.017760));
  /* To correct round-off error for T points in training set */
  /* t_feat = t_feat + 1e-4; */
  /* Make sure t_feat is not exactly 0 (this produces an edge case error */
  if(t_feat <= 0) t_feat = 1.0e-30;
  
  /* Set up feature DMatrix */
  float feats[6] = {t_feat, n_h_feat, q_lw_feat, q_hi_feat, q_hei_feat, q_cvi_feat};
  safe_xgboost(XGDMatrixCreateFromMat(feats, 1, 6, 0, &feat_dmatrix));
  
  /* Set up shape of the output */
  uint64_t const* out_shape;
  uint64_t out_dim;
  
  /* Configuration information about the booster */
  char const config[] =
    "{\"training\": false, \"type\": 0, "
    "\"iteration_begin\": 0, \"iteration_end\": 0, \"strict_shape\": false}";
  
  /* Get predictions for log CF, HF */
  safe_xgboost(XGBoosterPredictFromDMatrix(*cf_booster, feat_dmatrix, config, &out_shape, &out_dim, &log_cf));
  safe_xgboost(XGBoosterPredictFromDMatrix(*hf_booster, feat_dmatrix, config, &out_shape, &out_dim, &log_hf));

  *cfun = pow(10.0, *log_cf);
  *hfun = pow(10.0, *log_hf);

  /* Free the memory */
  XGDMatrixFree(feat_dmatrix);
}


/* Declare relevant variables */
float Tem, Hden, Plw, Ph1, Pg1, Pc6;
int ierr;
float alt;

int main(int argc, char *argv[]) {
  Hden = 1.0e-3;
  Plw = 2.11814e-13;
  Ph1 = 1.08928e-13;
  Pg1 = 2.76947e-14;
  Pc6 = 1.03070e-17;

  float cfun, hfun;
  BoosterHandle cf_booster, hf_booster;

  /* Set path to CF, HF models */
  const char *cf_model_path = "/nfs/turbo/lsa-cavestru/dbrobins/ml_chf/models/gh12_rates/all_data/CF_Z_0/trained_model.txt";
  const	char *hf_model_path = "/nfs/turbo/lsa-cavestru/dbrobins/ml_chf/models/gh12_rates/all_data/HF_Z_0/trained_model.txt";

  xgb_get_models(cf_model_path, hf_model_path, &cf_booster, &hf_booster);
  
  alt = 1.0;
  while(alt < 9.0) {
    Tem = pow(10.0, alt);
    /* printf("Before: %3.1f %10.6E %10.6E\n", alt, cfun, hfun); */
    xgb_get_chf(&Tem, &Hden, &Plw, &Ph1, &Pg1, &Pc6, &cf_booster, &hf_booster, &cfun, &hfun, &ierr);
    if (ierr != 0)
      printf("Error in evaluating XGBoost model");
    printf("%3.1f %10.6E %10.6E\n", alt, cfun, hfun);

    alt = alt + 0.1;
  }
  
  /*
  Tem = pow(10.0, 94);
  xgb_get_chf(&Tem, &Hden, &Plw, &Ph1, &Pg1, &Pc6, &cf_booster, &hf_booster, &cfun, &hfun, &ierr);
  printf("%f %10.6E %10.6E\n", Tem, cfun, hfun);
  */
  XGBoosterFree(cf_booster);
  XGBoosterFree(hf_booster);
  
  return 0;
}


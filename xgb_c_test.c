#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <xgboost/c_api.h>

#define safe_xgboost(call) { \
  int err = (call); \
  if (err != 0) { \
    fprintf(stderr, "%s:%d: error in %s: %s\n", __FILE__, __LINE__, #call, XGBGetLastError()); \
    exit(1); \
  } \
}

int main() {
  /* Create needed XGBoost objects */
  BoosterHandle booster;
  DMatrixHandle dmatrix;
  /* Set path to model */
  const char *model_path = "/nfs/turbo/lsa-cavestru/dbrobins/ml_chf/models/gh12_rates/all_data/CF_Z_0/trained_model.txt";

  /* Create the booster to read the data into */
  safe_xgboost(XGBoosterCreate(NULL, 0, &booster));
  /* Read in the model */
  safe_xgboost(XGBoosterLoadModel(booster, model_path));

  /* Set up the features to feed into the model */
  const float test_input[6] = {0.9, 0.3, 0.2, 0.7, 0.8, 0.1};
  /* Convert it into a DMatrix */
  safe_xgboost(XGDMatrixCreateFromMat(test_input, 1, 6, 0, &dmatrix));

  /* Set up shape of the output */
  uint64_t const* out_shape;
  uint64_t out_dim;
  float const* out_result = NULL;

  /* Configuration information about the booster */
  char const config[] =
    "{\"training\": false, \"type\": 0, "
    "\"iteration_begin\": 0, \"iteration_end\": 0, \"strict_shape\": false}";
  
  /* Get prediction */
  safe_xgboost(XGBoosterPredictFromDMatrix(booster, dmatrix, config, &out_shape, &out_dim, &out_result));

  /* Print to check */
  printf("Model prediction is %f\n", *out_result);

  /* Free the memory */
  XGBoosterFree(booster);
  XGDMatrixFree(dmatrix);

  return 0;
}


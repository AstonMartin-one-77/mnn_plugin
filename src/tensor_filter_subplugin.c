/**
 * GStreamer Tensor_Filter example Code
 * Copyright (C) 2019 MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 * This is a template with no license requirements.
 * Writers may alter the license to anything they want.
 * The author hereby allows to do so.
 */
/**
 * @file	tensor_filter_subplugin.c
 * @date	11 Oct 2019
 * @brief	NNStreamer tensor-filter subplugin template
 * @see		http://github.com/nnsuite/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs
 */

#include <string.h>
#include <glib.h>
#include <nnstreamer_plugin_api_filter.h>

void init_filter_example (void) __attribute__ ((constructor));
void fini_filter_example (void) __attribute__ ((destructor));

/**
 * @brief If you need to store session or model data,
 *        this is what you want to fill in.
 */
typedef struct
{
  gchar *model_path; /**< This is a sample. You may remove/change it */
} example_pdata;

static void example_close (const GstTensorFilterProperties * prop,
    void **private_data);

/**
 * @brief Check condition to reopen model.
 */
static int
example_reopen (const GstTensorFilterProperties * prop, void **private_data)
{
  /**
   * @todo Update condition to reopen model.
   *
   * When called the callback 'open' with user data,
   * check the model file or other condition whether the model or framework should be reloaded.
   * Below is example: when model file is changed, return 1 to reopen model.
   */
  example_pdata *pdata = *private_data;

  if (prop->num_models > 0 && strcmp (prop->model_files[0], pdata->model_path) != 0) {
    return 1;
  }

  return 0;
}

/**
 * @brief The standard tensor_filter callback
 */
static int
example_open (const GstTensorFilterProperties * prop, void **private_data)
{
  example_pdata *pdata;

  if (*private_data != NULL) {
    if (example_reopen (prop, private_data) != 0) {
      example_close (prop, private_data);  /* "reopen" */
    } else {
      return 1;
    }
  }

  pdata = g_new0 (example_pdata, 1);
  if (pdata == NULL)
    return -1;

  *private_data = (void *) pdata;

  /** @todo Initialize your own framework or hardware here */

  if (prop->num_models > 0)
    pdata->model_path = g_strdup (prop->model_files[0]);

  return 0;
}

/**
 * @brief The standard tensor_filter callback
 */
static void
example_close (const GstTensorFilterProperties * prop, void **private_data)
{
  example_pdata *pdata;
  pdata = *private_data;

  /** @todo Close what you have opened/allocated with example_open */

  g_free (pdata->model_path);
  pdata->model_path = NULL;

  g_free (pdata);
  *private_data = NULL;
}

/**
 * @brief The standard tensor_filter callback for static input/output dimension.
 * @note If you want to support flexible/dynamic input/output dimension,
 *       read nnstreamer_plugin_api_filter.h and supply the
 *       setInputDimension callback.
 */
static int
example_getInputDim (const GstTensorFilterProperties * prop,
    void **private_data, GstTensorsInfo * info)
{
  /** @todo Configure info with the proper (static) input tensor dimension */
  return 0;
}

/**
 * @brief The standard tensor_filter callback for static input/output dimension.
 * @note If you want to support flexible/dynamic input/output dimension,
 *       read nnstreamer_plugin_api_filter.h and supply the
 *       setInputDimension callback.
 */
static int
example_getOutputDim (const GstTensorFilterProperties * prop,
    void **private_data, GstTensorsInfo * info)
{
  /** @todo Configure info with the proper (static) output tensor dimension */
  return 0;
}

/**
 * @brief The standard tensor_filter callback
 */
static int
example_invoke (const GstTensorFilterProperties * prop,
    void **private_data, const GstTensorMemory * input,
    GstTensorMemory * output)
{
  /** @todo Call your framework/hardware with the given input. */
  return 0;
}

static gchar filter_subplugin_example[] = "example";

static GstTensorFilterFramework NNS_support_example = {
#ifdef GST_TENSOR_FILTER_API_VERSION_DEFINED
  .version = GST_TENSOR_FILTER_FRAMEWORK_V0,
#else
  .name = filter_subplugin_example,
  .allow_in_place = FALSE,
  .allocate_in_invoke = FALSE,
  .run_without_model = FALSE,
  .invoke_NN = example_invoke,
  .getInputDimension = example_getInputDim,
  .getOutputDimension = example_getOutputDim,
#endif
  .open = example_open,
  .close = example_close,
};

/**@brief Initialize this object for tensor_filter subplugin runtime register */
void
init_filter_example (void)
{
#ifdef GST_TENSOR_FILTER_API_VERSION_DEFINED
  NNS_support_example.name = filter_subplugin_example;
  NNS_support_example.allow_in_place = FALSE;
  NNS_support_example.allocate_in_invoke = FALSE;
  NNS_support_example.run_without_model = FALSE;
  NNS_support_example.invoke_NN = example_invoke;
  NNS_support_example.getInputDimension = example_getInputDim;
  NNS_support_example.getOutputDimension = example_getOutputDim;
#endif
  nnstreamer_filter_probe (&NNS_support_example);
}

/** @brief Destruct the subplugin */
void
fini_filter_example (void)
{
  nnstreamer_filter_exit (NNS_support_example.name);
}

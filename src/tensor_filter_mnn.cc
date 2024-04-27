/**
 * GStreamer Tensor_Filter MNN delegate
 * Copyright (C) 2019 MyungJoo Ham <myungjoo.ham@samsung.com>
 */
/**
 * @file   tensor_filter_mnn.cc
 * @date   22 Apr 2024
 * @brief  NNStreamer tensor-filter subplugin for MNN delegate
 * @see	   http://github.com/nnsuite/nnstreamer
 * @see    https://github.com/alibaba/MNN 
 * @author Georgii Zagoruiko <astonmartin-one-77@outlook.com>
 * @bug    No known bugs
 * 
 * This is the per-NN-framework plugin (MNN) for tensor_filter
 */

#include <functional>
#include <glib.h>
#include <nnstreamer_cppplugin_api_filter.hh>
#include <nnstreamer_log.h>
#include <nnstreamer_plugin_api_util.h>
#include <nnstreamer_util.h>
#include <thread>

#include <MNN/Interpreter.hpp>

/** First iteration: CPU support only */
namespace nnstreamer
{
namespace tensorFilter_mnn
{

G_BEGIN_DECLS
void init_filter_mnn (void) __attribute__ ((constructor));
void fini_filter_mnn (void) __attribute__ ((destructor));
G_END_DECLS

class mnn_subplugin final : public tensor_filter_subplugin
{
    public:
    static void init_filter_mnn ();
    static void fini_filter_mnn ();

    mnn_subplugin ();
    ~mnn_subplugin ();

    tensor_filter_subplugin &getEmptyInstance ();
    void configure_instance (const GstTensorFilterProperties *prop);
    void invoke (const GstTensorMemory *input, GstTensorMemory *output);
    void getFrameworkInfo (GstTensorFilterFrameworkInfo &info);
    int getModelInfo (model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info);
    int eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data);

    private:
    static const char *name;
    bool empty_model; /**< Empty (not initialized) model flag */
    static const GstTensorFilterFrameworkInfo info; /**< Framework info */
    GstTensorsInfo inputInfo; /**< Input tensors metadata */
    GstTensorsInfo outputInfo; /**< Output tensors metadata */    

    static mnn_subplugin *registeredRepresentation;

    Session* session;
    std::unique_ptr<MNN::Interpreter> interpreter;
    const std::map<std::string, MNN::Tensor*> &inputMap;
    const std::map<std::string, MNN::Tensor*> &outputMap;
};

/**
 * @brief Describe framework information.
 */
const GstTensorFilterFrameworkInfo mnn_subplugin::info = 
{ 
    .name = "alibaba/MNN",
    .allow_in_place = FALSE,
    .allocate_in_invoke = FALSE,
    .run_without_model = FALSE,
    .verify_model_path = TRUE,
    .hw_list = (const accl_hw[]){ ACCL_CPU },
    .num_hw = 1,
    .accl_auto = ACCL_CPU,
    .accl_default = ACCL_CPU,
    .statistics = nullptr
};

/**
 * @brief constructor of mnn_subplugin
 */
mnn_subplugin::mnn_subplugin () : tensor_filter_subplugin ()
{
    gst_tensors_info_init (std::addressof (inputInfo));
    gst_tensors_info_init (std::addressof (outputInfo));
}

/**
 * @brief destructor of mnn_subplugin
 */
mnn_subplugin::~mnn_subplugin ()
{
    gst_tensors_info_free (std::addressof (inputInfo));
    gst_tensors_info_free (std::addressof (outputInfo));

    if (empty_model)
        return;

    mnn_subplugin::interpreter->releaseModel ();
    mnn_subplugin::interpreter->releaseSession (mnn_subplugin::session);
    empty_model = true;
}

/**
 * @brief Get empty instance of mnn_subplugin
 */
tensor_filter_subplugin & mnn_subplugin::getEmptyInstance ()
{
    return *(new mnn_subplugin ());
}

/**
 * @brief Configure instance of mnn_subplugin
 */
void mnn_subplugin::configure_instance (const GstTensorFilterProperties *prop)
{
    gst_tensors_info_copy (std::addressof (inputInfo), std::addressof (prop->input_meta));
    gst_tensors_info_copy (std::addressof (outputInfo), std::addressof (prop->output_meta));

    mnn_subplugin::interpreter = std::unique_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile((prop->model_files[0]).c_str()));

    if (nullptr == interpreter)
        throw std::runtime_error ("MNN: model missing or invalid initialization");

    ScheduleConfig conf;
    mnn_subplugin::session = interpreter->createSession(conf);

    mnn_subplugin::inputMap = interpreter->getSessionInputAll(mnn_subplugin::session);
    mnn_subplugin::inputMap = interpreter->getSessionOutputAll(mnn_subplugin::session);
}

/**
 * @brief Invoke instance of mnn_subplugin
 */
void mnn_subplugin::invoke (const GstTensorMemory *input, GstTensorMemory *output)
{
    if (empty_model)
        throw std::runtime_error (
            "Model is empty: MNN instance is not configured and "
            "its \"invoke\" method is called. This may be an internal bug of "
            "nnstreamer or mnn-subplugin unless if you have directly accessed "
            "mnn-subplugin.");
    

}

/**
 * @brief Register instance of mnn_subplugin
 */
void mnn_subplugin::init_filter_mnn (void)
{
    registeredRepresentation = tensor_filter_subplugin::register_subplugin<mnn_subplugin> ();
}

/**
 * @brief Destruct the subplugin
 */
void mnn_subplugin::fini_filter_mnn (void)
{
    g_assert (registeredRepresentation != nullptr);
    tensor_filter_subplugin::unregister_subplugin (registeredRepresentation);
}

/**
 * @brief initializer
 */
void init_filter_mnn ()
{
    mnn_subplugin::init_filter_mnn ();
}

/**
 * @brief finalizer
 */
void fini_filter_mnn ()
{
    mnn_subplugin::fini_filter_mnn ();
}

} /* namespace tensorFilter_mnn */
}  /* namespace nnstreamer */

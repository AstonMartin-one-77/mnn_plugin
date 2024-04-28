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
    bool empty_model; /**< Empty (not initialized) model flag */
    static const char *name;
    static const GstTensorFilterFrameworkInfo info; /**< Framework info */
    GstTensorsInfo inputInfo; /**< Input tensors metadata */
    GstTensorsInfo outputInfo; /**< Output tensors metadata */  

    static mnn_subplugin *registeredRepresentation;

    Session* session;
    std::unique_ptr<MNN::Interpreter> interpreter;
    std::unique_ptr<std::map<std::string, MNN::Tensor*>> inputMap;
    std::unique_ptr<std::map<std::string, MNN::Tensor*>> outputMap;

    // std::unique_ptr<MNN::CV::ImageProcess> converter;

    void convertTensorInfo (std::map<std::string, MNN::Tensor*> &modelInfo, GstTensorsInfo &info);
};

/**
 * @brief Describe framework information
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

    mnn_subplugin::inputMap =  std::unique_ptr<std::map<std::string, MNN::Tensor*>>(interpreter->getSessionInputAll(mnn_subplugin::session));
    mnn_subplugin::outputMap = std::unique_ptr<std::map<std::string, MNN::Tensor*>>(interpreter->getSessionOutputAll(mnn_subplugin::session));

    if (inputInfo.num_tensors != inputMap->size())
        throw std::invalid_argument (
            std::string ("Wrong number of input tensor")
            + ": Found in argument = " + std::to_string (inputInfo.num_tensors)
            + ", Found in model file = " + std::to_string (inputMap->size ()));
    
    // int idx = 0;
    // for (auto const &itr : mnn_subplugin::inputMap) {
    //     GstTensorInfo &info = mnn_subplugin::inputInfo.info[idx++];
    //     for (int i = 0; info.dimension[i]; ++i) {
    //         if (itr.second->shape ()[i] != info.dimension[i])
    //             throw std::invalid_argument (
    //                 std::string ("Wrong shape of input tensor")
    //                 + "[" + itr.first + "]"
    //                 + ": Found in argument = " + std::to_string (info.dimension[i])
    //                 + ", Found in model file = " + std::to_string (itr.second->shape ()[i]));
    //     }
    // }

    if (outputInfo.num_tensors != outputMap->size())
        throw std::invalid_argument (
            std::string ("Wrong number of input tensor")
            + ": Found in argument = " + std::to_string (inputInfo.num_tensors)
            + ", Found in model file = " + std::to_string (inputMap->size ()));
    
    // idx = 0;
    // for (auto const &itr : mnn_subplugin::outputMap) {
    //     GstTensorInfo &info = mnn_subplugin::outputInfo.info[idx++];
    //     for (int i = 0; info.dimension[i]; ++i) {
    //         if (itr.second->shape ()[i] != info.dimension[i])
    //             throw std::invalid_argument (
    //                 std::string ("Wrong shape of output tensor")
    //                 + "[" + itr.first + "]"
    //                 + ": Found in argument = " + std::to_string (info.dimension[i])
    //                 + ", Found in model file = " + std::to_string (itr.second->shape ()[i]));
    //     }
    // }

    // MNN::CV::ImageProcess::Config img_cfg;
    // converter = std::unique_ptr<MNN::CV::ImageProcess>(MNN::CV::ImageProcess::create(img_cfg));

    empty_model = false;
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
    int inIdx = 0;
    for (auto &itr : mnn_subplugin::inputMap) {
        if (itr.second->size () != input[inIdx].size)
            throw std::invalid_argument (
                std::string ("Wrong data size of input tensor[") + itr->first + "]: " +
                + ": Found in argument = " + std::to_string (input[inIdx].size)
                + ", Found in model = " + std::to_string (itr.second->size ()));
        memcpy(itr.second->host (), input[inIdx++].data, itr.second->size ());
    }

    mnn_subplugin::interpreter->runSession (mnn_subplugin::session);

    int outIdx = 0;
    for (auto &itr : mnn_subplugin::outputMap) {
        if (itr.second->size () != output[outIdx].size)
            throw std::invalid_argument (
                std::string ("Wrong data size of output tensor[") + itr->first + "]: " +
                + ": Found in argument = " + std::to_string (input[outIdx].size)
                + ", Found in model = " + std::to_string (itr.second->size ()));
        memcpy(output[outIdx++].data, itr.second->host (), itr.second->size ());
    }
}

/**
 * @brief Register instance of mnn_subplugin
 */
void mnn_subplugin::init_filter_mnn (void)
{
    registeredRepresentation = tensor_filter_subplugin::register_subplugin<mnn_subplugin> ();
}

/**
 * @brief Destruct subplugin
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

/**
 * @brief Method to get the information of subplugin
 */
void mnn_subplugin::getFrameworkInfo (GstTensorFilterFrameworkInfo &info)
{
    info = mnn_subplugin::info;
}

/**
 * @brief Get MNN model information
 */
int mnn_subplugin::getModelInfo (model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info)
{
  switch (ops) {
    case GET_IN_OUT_INFO:
      gst_tensors_info_copy (std::addressof (in_info), std::addressof (inputInfo));
      gst_tensors_info_copy (std::addressof (out_info), std::addressof (outputInfo));
      break;
    case SET_INPUT_INFO:
    default:
      return -ENOENT;
  }
  return 0;
}

/**
 * @brief Method to handle events
 */
int mnn_subplugin::eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data)
{
    UNUSED (ops);
    UNUSED (data);
    return -ENOENT;
}

/**
 * @brief Convert MNN model info to GST Tensor info
 */
void mnn_subplugin::convertTensorInfo (std::map<std::string, MNN::Tensor*> &mnnInfo, GstTensorsInfo &info)
{
    gst_tensors_info_init (std::addressof (info));
    info.num_tensors = (unsigned int) mnnInfo->size();

    unsigned int idx = 0;
    for (auto &itr : mnnInfo) {
        GstTensorInfo *_info = gst_tensors_info_get_nth_info (std::addressof (info), idx++);

        switch(itr.second->getType().code) {
            case halide_type_int:
                if (64U == itr.second->getType().bits) {
                    _info->type = _NNS_INT64;
                } else if (32U == itr.second->getType().bits) {
                    _info->type = _NNS_INT32;
                } else if (16U == itr.second->getType().bits) {
                    _info->type = _NNS_INT16;
                } else if (8U == itr.second->getType().bits) {
                    _info->type = _NNS_INT8;
                } else {
                    throw std::invalid_argument (
                        std::string ("Usupported data width of MNN tensor[") + itr->first + "]: " +
                        + ", Found in model = " + std::to_string (itr.second->getType().bits));
                }
            break;
            case halide_type_uint:
                if (64U == itr.second->getType().bits) {
                    _info->type = _NNS_UINT64;
                } else if (32U == itr.second->getType().bits) {
                    _info->type = _NNS_UINT32;
                } else if (16U == itr.second->getType().bits) {
                    _info->type = _NNS_UINT16;
                } else if (8U == itr.second->getType().bits) {
                    _info->type = _NNS_UINT8;
                } else {
                    throw std::invalid_argument (
                        std::string ("Usupported data width of MNN tensor[") + itr->first + "]: " +
                        + ", Found in model = " + std::to_string (itr.second->getType().bits));
                }
            break;
            case halide_type_float:
                if (64U == itr.second->getType().bits) {
                    _info->type = _NNS_FLOAT64;
                } else if (32U == itr.second->getType().bits) {
                    _info->type = _NNS_FLOAT32;
                } else if (16U == itr.second->getType().bits) {
                    _info->type = _NNS_FLOAT16;
                } else {
                    throw std::invalid_argument (
                        std::string ("Usupported data width of MNN tensor[") + itr->first + "]: " +
                        + ", Found in model = " + std::to_string (itr.second->getType().bits));
                }
            break;
            // case halide_type_bfloat:
            // break;
            default:
                throw std::invalid_argument (
                    std::string ("Usupported data type of MNN tensor[") + itr->first + "]: " +
                    + ", Found in model = " + std::to_string (itr.second->getType().code));
            break;
        }

        if (itr.second->shape().size() > NNS_TENSOR_RANK_LIMIT)
            throw std::invalid_argument (
                    std::string ("Rank limit (NNStreamer) is excessed by MNN tensor[") + itr->first + "]: " +
                    + ", Found in model = " + std::to_string (itr.second->shape().size()));
        
        for (int i = 0; i < itr.second->shape().size(); ++i)
            _info->dimension[i] = itr.second->shape()[i]; /* TODO: investigate correct order of dim translation */

        for (int i = itr.second->shape(); i < NNS_TENSOR_RANK_LIMIT; ++i)
            _info->dimension[i] = 0; /* set to 0 all other dims */

        _info->name = g_strdup (itr.first);
    }
}

} /* namespace tensorFilter_mnn */
}  /* namespace nnstreamer */

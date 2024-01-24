#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct HealthRequest {}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct HealthResponse {}
/// / Empty request
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct InfoRequest {}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct InfoResponse {
    #[prost(bool, tag = "1")]
    pub requires_padding: bool,
    #[prost(string, tag = "2")]
    pub dtype: ::prost::alloc::string::String,
    #[prost(string, tag = "3")]
    pub device_type: ::prost::alloc::string::String,
    #[prost(uint32, optional, tag = "4")]
    pub window_size: ::core::option::Option<u32>,
}
/// / Empty request
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ServiceDiscoveryRequest {}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ServiceDiscoveryResponse {
    /// / Other shards urls
    #[prost(string, repeated, tag = "1")]
    pub urls: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ClearCacheRequest {
    /// / Optional batch id
    #[prost(uint64, optional, tag = "1")]
    pub id: ::core::option::Option<u64>,
}
/// / Empty response
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ClearCacheResponse {}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct NextTokenChooserParameters {
    /// / exponential scaling output probability distribution
    #[prost(float, tag = "1")]
    pub temperature: f32,
    /// / restricting to the k highest probability elements
    #[prost(uint32, tag = "2")]
    pub top_k: u32,
    /// / restricting to top tokens summing to prob_cut_off <= prob_cut_off
    #[prost(float, tag = "3")]
    pub top_p: f32,
    /// / restricting to top tokens summing to prob_cut_off <= prob_cut_off
    #[prost(float, tag = "4")]
    pub typical_p: f32,
    /// / apply sampling on the logits
    #[prost(bool, tag = "5")]
    pub do_sample: bool,
    /// / random seed for sampling
    #[prost(uint64, tag = "6")]
    pub seed: u64,
    /// / repetition penalty
    #[prost(float, tag = "7")]
    pub repetition_penalty: f32,
    /// / token watermarking using "A Watermark for Large Language Models"
    #[prost(bool, tag = "8")]
    pub watermark: bool,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct StoppingCriteriaParameters {
    /// / Maximum number of generated tokens
    #[prost(uint32, tag = "1")]
    pub max_new_tokens: u32,
    /// / Optional stopping sequences
    #[prost(string, repeated, tag = "2")]
    pub stop_sequences: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
    /// / Ignore end of sequence token
    /// / used for benchmarking
    #[prost(bool, tag = "3")]
    pub ignore_eos_token: bool,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Request {
    /// / Request ID
    #[prost(uint64, tag = "1")]
    pub id: u64,
    /// / The generation context
    #[prost(string, tag = "2")]
    pub inputs: ::prost::alloc::string::String,
    /// / Context truncation
    #[prost(uint32, tag = "3")]
    pub truncate: u32,
    /// / Next Token Chooser Parameters
    #[prost(message, optional, tag = "4")]
    pub parameters: ::core::option::Option<NextTokenChooserParameters>,
    /// / Stopping Criteria Parameters
    #[prost(message, optional, tag = "5")]
    pub stopping_parameters: ::core::option::Option<StoppingCriteriaParameters>,
    /// / Return prefill logprobs
    #[prost(bool, tag = "6")]
    pub prefill_logprobs: bool,
    /// / Return most likely n tokens
    #[prost(uint32, tag = "7")]
    pub top_n_tokens: u32,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Batch {
    /// / Batch ID
    #[prost(uint64, tag = "1")]
    pub id: u64,
    /// / Individual requests
    #[prost(message, repeated, tag = "2")]
    pub requests: ::prost::alloc::vec::Vec<Request>,
    /// / Batch size (==len(requests))
    #[prost(uint32, tag = "3")]
    pub size: u32,
    /// / Maximum number of tokens this batch will grow to
    #[prost(uint32, tag = "4")]
    pub max_tokens: u32,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CachedBatch {
    /// / Batch ID
    #[prost(uint64, tag = "1")]
    pub id: u64,
    /// / Individual requests ids
    #[prost(uint64, repeated, tag = "2")]
    pub request_ids: ::prost::alloc::vec::Vec<u64>,
    /// / Batch size (==len(requests))
    #[prost(uint32, tag = "3")]
    pub size: u32,
    /// / Maximum number of tokens this batch will grow to
    #[prost(uint32, tag = "4")]
    pub max_tokens: u32,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GeneratedText {
    /// / Output
    #[prost(string, tag = "1")]
    pub text: ::prost::alloc::string::String,
    /// / Number of generated tokens
    #[prost(uint32, tag = "2")]
    pub generated_tokens: u32,
    /// / Finish reason
    #[prost(enumeration = "FinishReason", tag = "3")]
    pub finish_reason: i32,
    /// / Seed
    #[prost(uint64, optional, tag = "4")]
    pub seed: ::core::option::Option<u64>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PrefillTokens {
    /// / Prefill Token IDs
    #[prost(uint32, repeated, tag = "1")]
    pub ids: ::prost::alloc::vec::Vec<u32>,
    /// / Prefill Logprobs
    #[prost(float, repeated, tag = "2")]
    pub logprobs: ::prost::alloc::vec::Vec<f32>,
    /// / Prefill tokens
    #[prost(string, repeated, tag = "3")]
    pub texts: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TopTokens {
    /// / Top Token IDs
    #[prost(uint32, repeated, tag = "1")]
    pub ids: ::prost::alloc::vec::Vec<u32>,
    /// / Top Logprobs
    #[prost(float, repeated, tag = "2")]
    pub logprobs: ::prost::alloc::vec::Vec<f32>,
    /// / Top Token Texts
    #[prost(string, repeated, tag = "3")]
    pub texts: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
    /// / If the tokens are special
    #[prost(bool, repeated, tag = "6")]
    pub is_special: ::prost::alloc::vec::Vec<bool>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Generation {
    /// / Request ID
    #[prost(uint64, tag = "1")]
    pub request_id: u64,
    /// / Prefill tokens (optional)
    #[prost(message, optional, tag = "2")]
    pub prefill_tokens: ::core::option::Option<PrefillTokens>,
    /// / Token ID
    #[prost(uint32, tag = "3")]
    pub token_id: u32,
    /// / Logprob
    #[prost(float, tag = "4")]
    pub token_logprob: f32,
    /// / Text
    #[prost(string, tag = "5")]
    pub token_text: ::prost::alloc::string::String,
    /// / Is it a special token
    #[prost(bool, tag = "6")]
    pub token_is_special: bool,
    /// / Complete generated text
    #[prost(message, optional, tag = "7")]
    pub generated_text: ::core::option::Option<GeneratedText>,
    /// / Top tokens
    #[prost(message, optional, tag = "8")]
    pub top_tokens: ::core::option::Option<TopTokens>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FilterBatchRequest {
    /// / Batch ID
    #[prost(uint64, tag = "1")]
    pub batch_id: u64,
    /// / Requests to keep
    #[prost(uint64, repeated, tag = "2")]
    pub request_ids: ::prost::alloc::vec::Vec<u64>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FilterBatchResponse {
    /// / Filtered Batch (cached)
    #[prost(message, optional, tag = "1")]
    pub batch: ::core::option::Option<CachedBatch>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PrefillRequest {
    /// / Batch
    #[prost(message, optional, tag = "1")]
    pub batch: ::core::option::Option<Batch>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PrefillResponse {
    /// / Generation
    #[prost(message, repeated, tag = "1")]
    pub generations: ::prost::alloc::vec::Vec<Generation>,
    /// / Next batch (cached)
    #[prost(message, optional, tag = "2")]
    pub batch: ::core::option::Option<CachedBatch>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DecodeRequest {
    /// / Cached batches
    #[prost(message, repeated, tag = "1")]
    pub batches: ::prost::alloc::vec::Vec<CachedBatch>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DecodeResponse {
    /// / Decodes
    #[prost(message, repeated, tag = "1")]
    pub generations: ::prost::alloc::vec::Vec<Generation>,
    /// / Next batch (cached)
    #[prost(message, optional, tag = "2")]
    pub batch: ::core::option::Option<CachedBatch>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WarmupRequest {
    /// / Batch to warmup on
    #[prost(message, repeated, tag = "1")]
    pub batches: ::prost::alloc::vec::Vec<Batch>,
}
/// / Empty response
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WarmupResponse {
    /// / Maximum number of tokens supported by the model
    #[prost(uint32, optional, tag = "1")]
    pub max_supported_total_tokens: ::core::option::Option<u32>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum FinishReason {
    Length = 0,
    EosToken = 1,
    StopSequence = 2,
}
impl FinishReason {
    /// String value of the enum field names used in the ProtoBuf definition.
    ///
    /// The values are not transformed in any way and thus are considered stable
    /// (if the ProtoBuf definition does not change) and safe for programmatic use.
    pub fn as_str_name(&self) -> &'static str {
        match self {
            FinishReason::Length => "FINISH_REASON_LENGTH",
            FinishReason::EosToken => "FINISH_REASON_EOS_TOKEN",
            FinishReason::StopSequence => "FINISH_REASON_STOP_SEQUENCE",
        }
    }
    /// Creates an enum from field names used in the ProtoBuf definition.
    pub fn from_str_name(value: &str) -> ::core::option::Option<Self> {
        match value {
            "FINISH_REASON_LENGTH" => Some(Self::Length),
            "FINISH_REASON_EOS_TOKEN" => Some(Self::EosToken),
            "FINISH_REASON_STOP_SEQUENCE" => Some(Self::StopSequence),
            _ => None,
        }
    }
}
/// Generated client implementations.
pub mod text_generation_service_client {
    #![allow(unused_variables, dead_code, missing_docs, clippy::let_unit_value)]
    use tonic::codegen::*;
    use tonic::codegen::http::Uri;
    #[derive(Debug, Clone)]
    pub struct TextGenerationServiceClient<T> {
        inner: tonic::client::Grpc<T>,
    }
    impl TextGenerationServiceClient<tonic::transport::Channel> {
        /// Attempt to create a new client by connecting to a given endpoint.
        pub async fn connect<D>(dst: D) -> Result<Self, tonic::transport::Error>
        where
            D: TryInto<tonic::transport::Endpoint>,
            D::Error: Into<StdError>,
        {
            let conn = tonic::transport::Endpoint::new(dst)?.connect().await?;
            Ok(Self::new(conn))
        }
    }
    impl<T> TextGenerationServiceClient<T>
    where
        T: tonic::client::GrpcService<tonic::body::BoxBody>,
        T::Error: Into<StdError>,
        T::ResponseBody: Body<Data = Bytes> + Send + 'static,
        <T::ResponseBody as Body>::Error: Into<StdError> + Send,
    {
        pub fn new(inner: T) -> Self {
            let inner = tonic::client::Grpc::new(inner);
            Self { inner }
        }
        pub fn with_origin(inner: T, origin: Uri) -> Self {
            let inner = tonic::client::Grpc::with_origin(inner, origin);
            Self { inner }
        }
        pub fn with_interceptor<F>(
            inner: T,
            interceptor: F,
        ) -> TextGenerationServiceClient<InterceptedService<T, F>>
        where
            F: tonic::service::Interceptor,
            T::ResponseBody: Default,
            T: tonic::codegen::Service<
                http::Request<tonic::body::BoxBody>,
                Response = http::Response<
                    <T as tonic::client::GrpcService<tonic::body::BoxBody>>::ResponseBody,
                >,
            >,
            <T as tonic::codegen::Service<
                http::Request<tonic::body::BoxBody>,
            >>::Error: Into<StdError> + Send + Sync,
        {
            TextGenerationServiceClient::new(InterceptedService::new(inner, interceptor))
        }
        /// Compress requests with the given encoding.
        ///
        /// This requires the server to support it otherwise it might respond with an
        /// error.
        #[must_use]
        pub fn send_compressed(mut self, encoding: CompressionEncoding) -> Self {
            self.inner = self.inner.send_compressed(encoding);
            self
        }
        /// Enable decompressing responses.
        #[must_use]
        pub fn accept_compressed(mut self, encoding: CompressionEncoding) -> Self {
            self.inner = self.inner.accept_compressed(encoding);
            self
        }
        /// Limits the maximum size of a decoded message.
        ///
        /// Default: `4MB`
        #[must_use]
        pub fn max_decoding_message_size(mut self, limit: usize) -> Self {
            self.inner = self.inner.max_decoding_message_size(limit);
            self
        }
        /// Limits the maximum size of an encoded message.
        ///
        /// Default: `usize::MAX`
        #[must_use]
        pub fn max_encoding_message_size(mut self, limit: usize) -> Self {
            self.inner = self.inner.max_encoding_message_size(limit);
            self
        }
        /// / Model Info
        pub async fn info(
            &mut self,
            request: impl tonic::IntoRequest<super::InfoRequest>,
        ) -> std::result::Result<tonic::Response<super::InfoResponse>, tonic::Status> {
            self.inner
                .ready()
                .await
                .map_err(|e| {
                    tonic::Status::new(
                        tonic::Code::Unknown,
                        format!("Service was not ready: {}", e.into()),
                    )
                })?;
            let codec = tonic::codec::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/generate.v1.TextGenerationService/Info",
            );
            let mut req = request.into_request();
            req.extensions_mut()
                .insert(GrpcMethod::new("generate.v1.TextGenerationService", "Info"));
            self.inner.unary(req, path, codec).await
        }
        /// / Service discovery
        pub async fn service_discovery(
            &mut self,
            request: impl tonic::IntoRequest<super::ServiceDiscoveryRequest>,
        ) -> std::result::Result<
            tonic::Response<super::ServiceDiscoveryResponse>,
            tonic::Status,
        > {
            self.inner
                .ready()
                .await
                .map_err(|e| {
                    tonic::Status::new(
                        tonic::Code::Unknown,
                        format!("Service was not ready: {}", e.into()),
                    )
                })?;
            let codec = tonic::codec::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/generate.v1.TextGenerationService/ServiceDiscovery",
            );
            let mut req = request.into_request();
            req.extensions_mut()
                .insert(
                    GrpcMethod::new(
                        "generate.v1.TextGenerationService",
                        "ServiceDiscovery",
                    ),
                );
            self.inner.unary(req, path, codec).await
        }
        /// / Empties batch cache
        pub async fn clear_cache(
            &mut self,
            request: impl tonic::IntoRequest<super::ClearCacheRequest>,
        ) -> std::result::Result<
            tonic::Response<super::ClearCacheResponse>,
            tonic::Status,
        > {
            self.inner
                .ready()
                .await
                .map_err(|e| {
                    tonic::Status::new(
                        tonic::Code::Unknown,
                        format!("Service was not ready: {}", e.into()),
                    )
                })?;
            let codec = tonic::codec::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/generate.v1.TextGenerationService/ClearCache",
            );
            let mut req = request.into_request();
            req.extensions_mut()
                .insert(
                    GrpcMethod::new("generate.v1.TextGenerationService", "ClearCache"),
                );
            self.inner.unary(req, path, codec).await
        }
        /// / Remove requests from a cached batch
        pub async fn filter_batch(
            &mut self,
            request: impl tonic::IntoRequest<super::FilterBatchRequest>,
        ) -> std::result::Result<
            tonic::Response<super::FilterBatchResponse>,
            tonic::Status,
        > {
            self.inner
                .ready()
                .await
                .map_err(|e| {
                    tonic::Status::new(
                        tonic::Code::Unknown,
                        format!("Service was not ready: {}", e.into()),
                    )
                })?;
            let codec = tonic::codec::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/generate.v1.TextGenerationService/FilterBatch",
            );
            let mut req = request.into_request();
            req.extensions_mut()
                .insert(
                    GrpcMethod::new("generate.v1.TextGenerationService", "FilterBatch"),
                );
            self.inner.unary(req, path, codec).await
        }
        /// / Warmup the model and compute max cache size
        pub async fn warmup(
            &mut self,
            request: impl tonic::IntoRequest<super::WarmupRequest>,
        ) -> std::result::Result<tonic::Response<super::WarmupResponse>, tonic::Status> {
            self.inner
                .ready()
                .await
                .map_err(|e| {
                    tonic::Status::new(
                        tonic::Code::Unknown,
                        format!("Service was not ready: {}", e.into()),
                    )
                })?;
            let codec = tonic::codec::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/generate.v1.TextGenerationService/Warmup",
            );
            let mut req = request.into_request();
            req.extensions_mut()
                .insert(GrpcMethod::new("generate.v1.TextGenerationService", "Warmup"));
            self.inner.unary(req, path, codec).await
        }
        /// / Prefill batch and decode first token
        pub async fn prefill(
            &mut self,
            request: impl tonic::IntoRequest<super::PrefillRequest>,
        ) -> std::result::Result<
            tonic::Response<super::PrefillResponse>,
            tonic::Status,
        > {
            self.inner
                .ready()
                .await
                .map_err(|e| {
                    tonic::Status::new(
                        tonic::Code::Unknown,
                        format!("Service was not ready: {}", e.into()),
                    )
                })?;
            let codec = tonic::codec::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/generate.v1.TextGenerationService/Prefill",
            );
            let mut req = request.into_request();
            req.extensions_mut()
                .insert(GrpcMethod::new("generate.v1.TextGenerationService", "Prefill"));
            self.inner.unary(req, path, codec).await
        }
        /// / Decode token for a list of prefilled batches
        pub async fn decode(
            &mut self,
            request: impl tonic::IntoRequest<super::DecodeRequest>,
        ) -> std::result::Result<tonic::Response<super::DecodeResponse>, tonic::Status> {
            self.inner
                .ready()
                .await
                .map_err(|e| {
                    tonic::Status::new(
                        tonic::Code::Unknown,
                        format!("Service was not ready: {}", e.into()),
                    )
                })?;
            let codec = tonic::codec::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/generate.v1.TextGenerationService/Decode",
            );
            let mut req = request.into_request();
            req.extensions_mut()
                .insert(GrpcMethod::new("generate.v1.TextGenerationService", "Decode"));
            self.inner.unary(req, path, codec).await
        }
        /// / Health check
        pub async fn health(
            &mut self,
            request: impl tonic::IntoRequest<super::HealthRequest>,
        ) -> std::result::Result<tonic::Response<super::HealthResponse>, tonic::Status> {
            self.inner
                .ready()
                .await
                .map_err(|e| {
                    tonic::Status::new(
                        tonic::Code::Unknown,
                        format!("Service was not ready: {}", e.into()),
                    )
                })?;
            let codec = tonic::codec::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/generate.v1.TextGenerationService/Health",
            );
            let mut req = request.into_request();
            req.extensions_mut()
                .insert(GrpcMethod::new("generate.v1.TextGenerationService", "Health"));
            self.inner.unary(req, path, codec).await
        }
    }
}

/// Single shard Client
use crate::pb::generate::v1::text_generation_service_client::TextGenerationServiceClient;
use crate::pb::generate::v1::*;
use crate::Result;
use grpc_metadata::InjectTelemetryContext;
use std::cmp::min;
use std::env;
use tonic::transport::{Channel, Uri};
use tracing::instrument;

/// Text Generation Inference gRPC client
#[derive(Debug, Clone)]
pub struct Client {
    stub: TextGenerationServiceClient<Channel>,
}

impl Client {
    /// Returns a client connected to the given url
    pub async fn connect(uri: Uri) -> Result<Self> {
        let channel = Channel::builder(uri).connect().await?;

        Ok(Self {
            stub: TextGenerationServiceClient::new(channel),
        })
    }

    /// Returns a client connected to the given unix socket
    pub async fn connect_uds(path: String) -> Result<Self> {
        let channel = Channel::from_shared("http://[::]:50051".to_string())
            .unwrap()
            .connect_with_connector(tower::service_fn(move |_: Uri| {
                tokio::net::UnixStream::connect(path.clone())
            }))
            .await?;

        Ok(Self {
            stub: TextGenerationServiceClient::new(channel),
        })
    }

    /// Returns a list of uris or unix sockets of all shards
    #[instrument(skip(self))]
    pub async fn service_discovery(&mut self) -> Result<Vec<String>> {
        let request = tonic::Request::new(ServiceDiscoveryRequest {}).inject_context();
        let response = self.stub.service_discovery(request).await?;
        let urls = response
            .into_inner()
            .urls
            .into_iter()
            // Remove unix socket prefix
            .map(|url| match url.strip_prefix("unix://") {
                None => url,
                Some(stripped_url) => stripped_url.to_string(),
            })
            .collect();
        Ok(urls)
    }

    /// Get model info
    #[instrument(skip(self))]
    pub async fn info(&mut self) -> Result<InfoResponse> {
        let request = tonic::Request::new(InfoRequest {}).inject_context();
        let response = self.stub.info(request).await?.into_inner();
        Ok(response)
    }

    /// Get model health
    #[instrument(skip(self))]
    pub async fn health(&mut self) -> Result<HealthResponse> {
        let request = tonic::Request::new(HealthRequest {}).inject_context();
        let response = self.stub.health(request).await?.into_inner();
        Ok(response)
    }

    /// Clear the past generations cache
    #[instrument(skip(self))]
    pub async fn clear_cache(&mut self, batch_id: Option<u64>) -> Result<()> {
        let request = tonic::Request::new(ClearCacheRequest { id: batch_id }).inject_context();
        self.stub.clear_cache(request).await?;
        Ok(())
    }

    /// Filter a cached batch
    #[instrument(skip(self))]
    pub async fn filter_batch(
        &mut self,
        batch_id: u64,
        request_ids: Vec<u64>,
    ) -> Result<Option<CachedBatch>> {
        let request = tonic::Request::new(FilterBatchRequest {
            batch_id,
            request_ids,
        })
        .inject_context();
        let filtered_batch = self.stub.filter_batch(request).await?.into_inner();
        Ok(filtered_batch.batch)
    }

    /// Warmup on a max size batch
    ///
    /// Returns the maximum amount of tokens supported by the hardware
    #[instrument(skip_all)]
    pub async fn warmup(
        &mut self,
        max_input_length: u32,
        max_prefill_tokens: u32,
        max_total_tokens: u32,
    ) -> Result<Option<u32>> {
        let mut n_tokens = 0;
        let mut req_id = 0;
        let mut requests = Vec::new();
        let skip_tokenizer_in_tgi = env::var("SKIP_TOKENIZER_IN_TGI").ok().map_or(false, |value| value.to_lowercase() == "true");
        // Create requests
        while n_tokens < max_prefill_tokens {
            let truncate = min(max_input_length, max_prefill_tokens - n_tokens);
            let inputs = if skip_tokenizer_in_tgi {
                "1, 1, 518, 25580, 29962, 3532, 14816, 29903, 6778, 13, 3492, 526, 385, 319, 29902, 
                20255, 393, 6911, 2305, 1284, 2472, 29889, 4911, 674, 366, 2367, 366, 263, 1139, 29889, 
                3575, 3414, 338, 304, 1234, 408, 10847, 3730, 408, 366, 508, 29889, 5806, 22862, 1348, 
                4331, 29899, 29890, 858, 1022, 322, 26922, 596, 1234, 29889, 13, 29966, 829, 14816, 
                29903, 6778, 13, 13, 29954, 5428, 278, 10541, 376, 29909, 6114, 411, 263, 6534, 29891, 
                260, 16234, 29877, 373, 902, 1250, 338, 19436, 263, 3708, 344, 411, 263, 2654, 1652, 
                11251, 1596, 1213, 508, 591, 17668, 393, 376, 1576, 6114, 29915, 29879, 3708, 344, 
                756, 2654, 18281, 373, 372, 1213, 29973, 13, 5856, 29901, 13, 29899, 4874, 13, 29899, 
                372, 338, 451, 1950, 304, 2649, 13, 29899, 694, 2567, 29892, 1235, 29915, 29879, 367, 
                16232, 408, 1950, 29889, 3834, 7291, 937, 29901, 518, 29914, 25580, 29962, 29871".to_string()
            } else {
                "_test ".to_string().repeat(max_input_length as usize)
            };
            requests.push(Request {
                id: req_id,
                // We truncate the input on the server side to be sure that it has the correct size
                inputs: inputs,
                truncate,
                // Set sampling parameters to also take these ops into account in the max memory
                parameters: Some(NextTokenChooserParameters {
                    temperature: 1.0,
                    top_k: 0,
                    top_p: 1.0,
                    typical_p: 1.0,
                    do_sample: false,
                    seed: 0,
                    repetition_penalty: 1.0,
                    watermark: false,
                }),
                stopping_parameters: Some(StoppingCriteriaParameters {
                    max_new_tokens: max_total_tokens - truncate,
                    stop_sequences: vec![],
                    ignore_eos_token: true,
                }),
                prefill_logprobs: false,
                top_n_tokens: 0,
            });
            n_tokens += max_input_length;
            req_id += 1;
        }

        let requests_new = requests.clone();

        let batch1 = Batch {
            id: 0,
            size: requests.len() as u32,
            requests,
            max_tokens: 0,
        };

        let batch2 = Batch {
            id: 1,
            size: requests_new.len() as u32,
            requests: requests_new,
            max_tokens: 0,
        };

        let mut batches = Vec::new();
        batches.push(batch1);
        batches.push(batch2);
        let request = tonic::Request::new(WarmupRequest { batches }).inject_context();
        let response = self.stub.warmup(request).await?.into_inner();
        Ok(response.max_supported_total_tokens)
    }

    /// Generate one token for each request in the given batch
    ///
    /// Returns Generation for each request in batch
    /// and the next cached batch
    #[instrument(skip_all, fields(id = &batch.id, size = &batch.size))]
    pub async fn prefill(
        &mut self,
        batch: Batch,
    ) -> Result<(Vec<Generation>, Option<CachedBatch>)> {
        let request = tonic::Request::new(PrefillRequest { batch: Some(batch) }).inject_context();
        let response = self.stub.prefill(request).await?.into_inner();
        Ok((response.generations, response.batch))
    }

    /// Generate one token for each request in the given cached batches
    ///
    /// Returns Generation for each request in batches
    /// and the next cached batch
    #[instrument(skip_all, fields(size = batches.iter().map(|batch|{batch.size}).sum::<u32>()))]
    pub async fn decode(
        &mut self,
        batches: Vec<CachedBatch>,
    ) -> Result<(Vec<Generation>, Option<CachedBatch>)> {
        let request = tonic::Request::new(DecodeRequest { batches }).inject_context();
        let response = self.stub.decode(request).await?.into_inner();
        Ok((response.generations, response.batch))
    }
}

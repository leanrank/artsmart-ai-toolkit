job "shotsmart" {
  datacenters = ["shotsmart"]
  type = "service"

  group "shotsmart" {
    count = 1

    update {
      max_parallel = 1
      min_healthy_time = "10s"
      healthy_deadline = "5m"
      progress_deadline = "10m"
      auto_revert = true
    }

    network {
			mode = "bridge"
			port "http" {
				to = 5000
      }
    }
		
		// TODO: add volume config?
    // it might be better to not use a volume since we save the model on S3 and download it on startup or when it's missing

		service {
			name = "shotsmart"
			port = "5000"
			
			check {
				type = "http"
				path = "/health-check"
				interval = "40s"
				timeout = "30s"
				expose = true
			}

			connect {
				sidecar_service {}
			}
    }
	
		task "shotsmart" {
			vault {
				policies = ["default_nomad_job"]
				change_mode = "signal"
				change_signal = "SIGUSR2"
			}
			// NOTE: since we don't add the volume in the service block. So I don't think we should define volume here

			template {
				data = <<-EOF
					{{ with secret "aws/creds/artsmart-engine" }}
            AWS_ACCESS_KEY_ID = "{{ .Data.access_key }}"
            AWS_SECRET_ACCESS_KEY = "{{ .Data.secret_key }}"
          {{ end }}

					{{ with secret "kv/artsmart-engine" }}
            AWS_S3_BUCKET_NAME = "{{ .Data.data.AWS_S3_BUCKET_NAME }}"
            QUEUE_WORKERS      = "{{ .Data.data.QUEUE_WORKERS }}"
            REPLICATE_API_TOKEN = "{{ .Data.data.REPLICATE_API_TOKEN }}"
            REPLICATE_CONTROLNET_MODEL_NAME = "{{ .Data.data.REPLICATE_CONTROLNET_MODEL_NAME }}"
            REPLICATE_DREAMBOOTH_MODEL_NAME = "{{ .Data.data.REPLICATE_DREAMBOOTH_MODEL_NAME }}"
            REPLICATE_LORA_MODEL_NAME = "{{ .Data.data.REPLICATE_LORA_MODEL_NAME }}"
            SQS_QUEUE_NAME = "{{ .Data.data.SQS_QUEUE_NAME }}"
          {{ end }}
        EOF
				env = true
				destination = "${NOMAD_SECRETS_DIR}/secrets.env"
			}

			env {
        PORT = "${NOMAD_PORT_http}"
        NODE_IP = "${NOMAD_IP_http}"
        AWS_DEFAULT_REGION = "us-east-1"
      }

			driver = "docker"

			config {
				image = "artsmartdev/artsmart-flux-trainer:[[ env "GIT_COMMIT_SHORT" ]]"

				auth {
					username = "[[ env "DOCKER_HUB_USERNAME" ]]"
          password = "[[ env "DOCKER_HUB_PASSWORD" ]]"
				}
			}

			resources {
				cpu = 8192
				memory = 100240

				# an Nvidia GPU with >=16GiB memory, preferable a A-100
        device "nvidia/gpu" {
          count = 1
          constraint {
            attribute = "${device.attr.memory}"
            operator  = ">="
            value     = "16 GiB"
          }

          affinity {
            attribute = "${device.model}"
            operator  = "regexp"
            value     = "A100"
          }
        }
			}
		}
  }
}
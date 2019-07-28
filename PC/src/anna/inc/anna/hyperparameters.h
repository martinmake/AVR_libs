#ifndef _ANNA_HYPERPARAMETERS_H_
#define _ANNA_HYPERPARAMETERS_H_

#include <inttypes.h>

namespace Anna
{
	struct Hyperparameters
	{
		private:
			float    m_dropout;
			float    m_lambda1;
			float    m_lambda2;
			float    m_learning_rate;
			uint8_t  m_batch_size;
			float    m_noise;

		public:
			Hyperparameters(void);
			~Hyperparameters(void);

		public:
			bool is_valid(void) const;

		public: // GETTERS
			float    dropout                       (void) const;
			float    lambda1                       (void) const;
			float    lambda2                       (void) const;
			float    learning_rate                 (void) const;
			uint8_t  batch_size                    (void) const;
			float    noise                         (void) const;
			uint16_t weight_generation_seed        (void) const;
			float    weight_generation_lower_limit (void) const;
			float    weight_generation_upper_limit (void) const;
		public: // SETTERS
			void channel_count (uint16_t new_channel_count );
			void dropout                       (float    new_dropout                      );
			void lambda1                       (float    new_lambda1                      );
			void lambda2                       (float    new_lambda2                      );
			void learning_rate                 (float    new_learning_rate                );
			void batch_size                    (uint8_t  new_batch_size                   );
			void noise                         (float    new_noise                        );

	};

	// [IN .CPP] bool Hyperparameters::is_valid(void) const { return TODO; }

	// GETTERS
	inline float    Hyperparameters::dropout                       (void) const { return m_dropout;                       }
	inline float    Hyperparameters::lambda1                       (void) const { return m_lambda1;                       }
	inline float    Hyperparameters::lambda2                       (void) const { return m_lambda2;                       }
	inline float    Hyperparameters::learning_rate                 (void) const { return m_learning_rate;                 }
	inline uint8_t  Hyperparameters::batch_size                    (void) const { return m_batch_size;                    }
	inline float    Hyperparameters::noise                         (void) const { return m_noise;                         }

	// SETTERS
	inline void Hyperparameters::dropout                       (float    new_dropout                      ) { m_dropout                       = new_dropout;                       }
	inline void Hyperparameters::lambda1                       (float    new_lambda1                      ) { m_lambda1                       = new_lambda1;                       }
	inline void Hyperparameters::lambda2                       (float    new_lambda2                      ) { m_lambda2                       = new_lambda2;                       }
	inline void Hyperparameters::learning_rate                 (float    new_learning_rate                ) { m_learning_rate                 = new_learning_rate;                 }
	inline void Hyperparameters::batch_size                    (uint8_t  new_batch_size                   ) { m_batch_size                    = new_batch_size;                    }
	inline void Hyperparameters::noise                         (float    new_noise                        ) { m_noise                         = new_noise;                         }
}

#endif

#ifndef _ANNA_DATASET_BASE_H_
#define _ANNA_DATASET_BASE_H_

#include <inttypes.h>
#include <assert.h>

#include "anna/shape.h"
#include "anna/tensor.h"

namespace Anna
{
	namespace Dataset
	{
		struct Item final
		{
			std::vector<float> input;
			std::vector<float> desired_output;
		};

		class Base
		{
			protected: // MEMBER VARIABLES
				std::vector<Item> m_training_items;
				std::vector<Item> m_testing_items;
				std::vector<Item> m_unlabeled_items;

			public: // CONSTRUCTORS AND DESTRUCTOR
				Base(void);
				~Base(void);

			protected:
				virtual void load(const std::string& path, std::vector<Item>& destination_items);

			public: // GETTERS
				const std::vector<Item>& training_items (void) const;
				const std::vector<Item>& testing_items  (void) const;
				const std::vector<Item>& unlabeled_items(void);
		};

		inline void Base::load(const std::string& path, std::vector<Item>& destination_items) { (void) path; (void) destination_items; assert(false && "THIS IS JUST AN INTERFACE"); }

		// GETTERS
		inline const std::vector<Item>& Base::training_items (void) const { return m_training_items;  }
		inline const std::vector<Item>& Base::testing_items  (void) const { return m_testing_items;   }
		inline const std::vector<Item>& Base::unlabeled_items(void)       { return m_unlabeled_items; }
	}
}

#endif

#ifndef _GPL_PRIMITIVE_CONTAINER_H_
#define _GPL_PRIMITIVE_CONTAINER_H_

#include "gpl/core.h"
#include "gpl/primitives/base.h"

namespace Gpl
{
	namespace Primitive
	{
		class Container : public Primitive::Base
		{
			public: // CONSTRUCTORS
				Container(const Gra::Math::vec2<unsigned int>& initial_position, const Gra::Math::vec2<unsigned int>& initial_size);

			public: // FUNCTIONS
				void draw(const Gra::Math::vec2<unsigned int>& resolution, const glm::mat4& parent_mvp) override;

			public: // GETTERS
				const Gra::Math::vec2<unsigned int>& position(void) const;
				const Gra::Math::vec2<unsigned int>& size    (void) const;
			public: // SETTERS
				void position(const Gra::Math::vec2<unsigned int>& new_position);
				void size    (const Gra::Math::vec2<unsigned int>& new_size    );

			private:
				Gra::Math::vec2<unsigned int> m_position;
				Gra::Math::vec2<unsigned int> m_size;

				DECLARATION_MANDATORY(Container)

		};

		// GETTERS
		DEFINITION_DEFAULT_GETTER(Container, position, const Gra::Math::vec2<unsigned int>&)
		DEFINITION_DEFAULT_GETTER(Container, size,     const Gra::Math::vec2<unsigned int>&)
		// SETTERS
		DEFINITION_DEFAULT_SETTER(Container, position, const Gra::Math::vec2<unsigned int>&)
		DEFINITION_DEFAULT_SETTER(Container, size,     const Gra::Math::vec2<unsigned int>&)

		DEFINITION_MANDATORY(Container, other.m_position, other.m_size)
	}
}

#endif

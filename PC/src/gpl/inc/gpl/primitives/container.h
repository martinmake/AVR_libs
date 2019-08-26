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
				Container(const Position& initial_position, const Size& initial_size);

			public: // FUNCTIONS
				void draw(std::queue<std::pair<Primitive::Container&, Data::Draw>>& queue);
				// void on_mouse_over(Data::MouseOver& data) override;

			public: // GETTERS
				const Position& position(void) const;
				const Size    & size    (void) const;

				std::vector<std::unique_ptr<Primitive::Base>>& primitives(void);

				bool colides(const Position& position) const override;
				bool is_container(void) const override;
			public: // SETTERS
				void position(const Gra::Math::vec2<unsigned int>& new_position);
				void size    (const Gra::Math::vec2<unsigned int>& new_size    );

			public: // OPERATORS
				template <typename T> Primitive::Base& operator<<(const T&  primitive);
				template <typename T> Primitive::Base& operator<<(      T&& primitive);
				const Primitive::Base& operator[](uint16_t index) const;
				      Primitive::Base& operator[](uint16_t index);

			private:
				Position m_position;
				Size     m_size;
				std::vector<std::unique_ptr<Primitive::Base>> m_primitives;

				DECLARATION_MANDATORY(Container)

		};

		// GETTERS
		DEFINITION_DEFAULT_GETTER(Container, position, const Position&)
		DEFINITION_DEFAULT_GETTER(Container, size,     const Size    &)
		inline std::vector<std::unique_ptr<Primitive::Base>>& Container::primitives(void) { return m_primitives; }
		inline bool Container::is_container(void) const { return true; }
		// SETTERS
		DEFINITION_DEFAULT_SETTER(Container, position, const Position&)
		DEFINITION_DEFAULT_SETTER(Container, size,     const Size    &)

		// OPERATORS
		template <typename T> Primitive::Base& Container::operator<<(const T&  primitive) { m_primitives.push_back(std::make_unique<T>(          primitive )); return *this; }
		template <typename T> Primitive::Base& Container::operator<<(      T&& primitive) { m_primitives.push_back(std::make_unique<T>(std::move(primitive))); return *this; }
		inline const Primitive::Base& Container::operator[](uint16_t index) const { return *m_primitives[index]; }
		inline       Primitive::Base& Container::operator[](uint16_t index)       { return *m_primitives[index]; }

		DEFINITION_MANDATORY(Container, other.m_position, other.m_size)
	}
}

#endif

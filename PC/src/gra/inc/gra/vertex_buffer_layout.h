#ifndef _GRA_VERTEX_BUFFER_LAYOUT_H_
#define _GRA_VERTEX_BUFFER_LAYOUT_H_

#include <vector>
#include <inttypes.h>

#include "gra/glstd.h"
#include "gra/gldebug.h"


namespace Gra
{
	struct VertexBufferLayoutElement
	{
		unsigned int  type;
		unsigned int  count;
		unsigned char normalized;
	};

	class VertexBufferLayout
	{
		private:
			std::vector<VertexBufferLayoutElement> m_elements;
			unsigned int m_stride;

		public:
			VertexBufferLayout(void);
			~VertexBufferLayout(void);

		public:
			template <typename T>
			void push(uint32_t count);

		// GETTERS
		public:
			const std::vector<VertexBufferLayoutElement>& elements(void) const;
			unsigned int stride(void) const;
	};

	template <typename T>
	void VertexBufferLayout::push(uint32_t count)
	{
		(void) count;
		static_assert(sizeof(T) && false, "TYPE UNAVAIBLE!");
	}
	template <>
	void VertexBufferLayout::push<float>(uint32_t count);
	template <>
	void VertexBufferLayout::push<unsigned int>(uint32_t count);
	template <>
	void VertexBufferLayout::push<unsigned char>(uint32_t count);

	// GETTERS
	inline const std::vector<VertexBufferLayoutElement>& VertexBufferLayout::elements(void) const
	{
		return m_elements;
	}
	inline unsigned int VertexBufferLayout::stride(void) const
	{
		return m_stride;
	}
}

#endif

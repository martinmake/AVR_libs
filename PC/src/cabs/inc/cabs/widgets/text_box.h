#ifndef _CABS_WIDGETS_TEXT_BOX_H_
#define _CABS_WIDGETS_TEXT_BOX_H_

#include <vector>

#include "cabs/widget.h"
#include "cabs/padding.h"

class TextBox : public Widget
{
	private:
		std::string m_text;
		Padding     m_padding;

	public:
		TextBox(void);
		~TextBox(void);

	protected:
		void draw_inside(void) const override;

	// GETTERS
	public:
		const std::string& text   (void) const;
		const Padding&     padding(void) const;

	// SETTERS
	public:
		void text   (const std::string& new_text   );
		void padding(const Padding&     new_padding);
};

// GETTERS
inline const std::string& TextBox::text(void) const
{
	return m_text;
}
inline const Padding& TextBox::padding(void) const
{
	return m_padding;
}

// GETTERS
inline void TextBox::text(const std::string& new_text)
{
	m_text = new_text;
}
inline void TextBox::padding(const Padding& new_padding)
{
	m_padding = new_padding;
}

#endif

#ifndef _CABS_WIDGETS_TEXT_BOX_H_
#define _CABS_WIDGETS_TEXT_BOX_H_

#include <vector>

#include "cabs/widget.h"

class TextBox : public Widget
{
	private:
		std::vector<std::string> m_text;

	public:
		TextBox(void);
		~TextBox(void);

	public:
		void draw(void) const override;

	// GETTERS
	public:
		const std::vector<std::string>& text(void) const;

	// SETTERS
	public:
		void text(const std::vector<std::string>& new_text);
};

// GETTERS
inline const std::vector<std::string>& TextBox::text(void) const
{
	return m_text;
}

// GETTERS
inline void TextBox::text(const std::vector<std::string>& new_text)
{
	m_text = new_text;
}

#endif

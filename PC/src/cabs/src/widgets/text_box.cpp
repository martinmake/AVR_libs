#include "cabs/widgets/text_box.h"

TextBox::TextBox(void)
{
}

TextBox::~TextBox(void)
{
}

void TextBox::draw(void) const
{
	Widget::draw();

	for (size_t i = 0; i < m_text.size(); i++)
	{
		if (m_text[i].size() > (unsigned) m_size.w() - 6)
		{
			mvwaddnstr(m_win_border, 2 + i, 3, m_text[i].c_str(), m_size.w() - 7);
			waddstr(m_win_border, "...");
		} else
			mvwaddstr(m_win_border, 2 + i, 3, m_text[i].c_str());
	}

	wrefresh(m_win_border);
}

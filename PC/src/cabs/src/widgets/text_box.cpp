#include "cabs/widgets/text_box.h"
#include "cabs/application.h"

TextBox::TextBox(void)
{
}

TextBox::~TextBox(void)
{
}

void TextBox::draw_inside(void) const
{
	wmove(m_win_border, 1 + m_padding.top(), 1 + m_padding.left());
	for (std::string::const_iterator it = m_text.begin(); it != m_text.end(); it++)
	{
		if
		(
			(*it == '<' && (it + 1 != m_text.end() || *(it + 1) == '!'))        & // is tag start
			(it == m_text.begin() || *(it - 1) != '\\')                         & // is tag not escaped
			(m_text.find_first_of('>', it - m_text.begin()) != std::string::npos) // is tag ended
		)
		{
			it += 2;
			std::string tag;
			for (; *it != '>'; it++)
				tag.push_back(*it);

			int tag_attr = Cabs::parse_tag(tag);

			wattrset(m_win_border, tag_attr);

			it++;
		}

		if (*it == '\n')
			wmove(m_win_border, 1 + getcury(m_win_border), 1 + m_padding.left());
		else
		{
			bool is_out_of_right_bound = getcurx(m_win_border) > m_size.w() - 2 - m_padding.right();
			if (is_out_of_right_bound)
				wmove(m_win_border, 1 + getcury(m_win_border), 1 + m_padding.left());

			waddch(m_win_border, *it);
		}

		bool is_out_of_bottom_bound = getcury(m_win_border) > m_size.h() - 2 - m_padding.bottom();
		if (is_out_of_bottom_bound)
		{
			wmove(m_win_border, getcury(m_win_border) - 1, 1);
			wclrtoeol(m_win_border);
			wmove(m_win_border, getcury(m_win_border), m_padding.left() + (m_size.w() - m_padding.left() - m_padding.right()) / 2 - 1);
			waddstr(m_win_border, "...");
			break;
		}
	}
}

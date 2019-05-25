#ifndef _CABS_SIZE_H_
#define _CABS_SIZE_H_

class Size
{
	private:
		int m_w = 0;
		int m_h = 0;

	public:
		Size(int initial_w, int initial_h);
		Size(void);
		~Size(void);

	// GETTERS
	public:
		int w(void) const;
		int h(void) const;

	// SETTERS
	public:
		void w(int new_w);
		void h(int new_h);
};

// Size GETTERS
inline int Size::w(void) const
{
	return m_w;
}
inline int Size::h(void) const
{
	return m_h;
}

// Size SETTERS
inline void Size::w(int new_w)
{
	m_w = new_w;
}
inline void Size::h(int new_h)
{
	m_h = new_h;
}

#endif

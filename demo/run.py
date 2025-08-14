import IdeaSearch_fit
from IdeaSearch import IdeaSearcher
from IdeaSearch_fit.miscellaneous import print_helloworld


def main():
    
    print_helloworld()
    
    IdeaSearch_fit.set_language("en")
    
    print_helloworld()
    
    IdeaSearch_fit.set_language("zh")
    
    print_helloworld()
    
    
if __name__ == "__main__":
    
    main()
    
    
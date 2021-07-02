
import pygame, sys, random, neat, numpy

global gravity, jump, gap
gravity = 0.05
jump = 4
gap = 200
screen=0
tmp = 0

#class for character
class Character:
    def __init__(self):
        self.char = pygame.image.load("assets/img/kurba.png").convert()
        self.char = pygame.transform.scale2x(self.char)
        self.char.set_colorkey((0,0,0))
        self.x_position = 100
        self.y_position = random.randint(100,700)
        self.collider = self.char.get_rect(center = (self.x_position,self.y_position))
        self.movement = 0
        self.score = 0
        
    def jump(self):
        self.movement = 0
        self.movement -= jump





#function to keep and display score
def score(s,screen):
    score_surface = pygame.font.Font(pygame.font.get_default_font(),50).render(str(s//2),True,(255,255,255))
    score_rect = score_surface.get_rect(center = (500,100))
    screen.blit(score_surface,score_rect)


#platform functions
def create_platform(platform):
    r = random.randint(250,650)
    bottom = platform.get_rect(midtop = (1800,r))
    top = platform.get_rect(midbottom = (1800,r - gap))
    return bottom, top

def move_platform(l):
    for i in l:
        i.centerx -= 5
    return l

def draw_platform(l,screen,platform):
    for i in l:
        if i.bottom > 720:
            screen.blit(platform,i)
        else:
            screen.blit(pygame.transform.flip(platform, False, True),i)




def main(genomes,config):
    #game start, window parameter and assets
   
    pygame.init()
    screen = pygame.display.set_mode((1280,720))
    clock = pygame.time.Clock()
    
    bg_surface = pygame.image.load("assets/img/background.png")
    
    
    
    
    platform = pygame.image.load("assets/img/kurba_platform.png").convert()
    platform = pygame.transform.scale2x(platform)
    
    platform.set_colorkey((0,0,0))
    
    
    #platform spawn
    SPAWNPLATFORM = pygame.USEREVENT
    pygame.time.set_timer(SPAWNPLATFORM, 1500)
    platform_list = []
    
    
    game_active = True
    
    
    players = [] #-> its players now
    networks = [] #-> networks
    genes = [] #-> gens of neat
    
    for _, g in genomes: #->loops g times first variable is index
        networks.append(neat.nn.FeedForwardNetwork.create(g,config)) #->network for every g
        players.append(Character()) #->ai players
        g.fitness = 0 #->fitness will be zero because its initilazation
        genes.append(g) #->add genes
        
    #main = Character()
    
       
        
    
    while len(players) > 0:
        for main in players:#->check for every player
            
                
            bird_index = players.index(main)
            genes[bird_index].fitness += 0.1 #->fitness boost to stay alive
            
           
            
            
            if len(platform_list)>0: #if there is a lenghth of platform then use neural network
                output = networks[bird_index].activate((main.y_position,numpy.linalg.norm(main.y_position - platform_list[0].centerx),numpy.linalg.norm(main.y_position - platform_list[0].x)))
                if output[0] > 0.25:
                    main.jump()
            
            #EVENT
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        main.jump()
                        
                        """if game_active==False:
                            game_active= True
                            main.score= 0
                            platform_list.clear()"""
                    
                if event.type == SPAWNPLATFORM:
                    platform_list.extend(create_platform(platform))
                   
            #SURFACE   
            screen.blit(bg_surface,(0,0))
            
            if game_active:
                
                #PLATFORM SPAWN
                platform_list = move_platform(platform_list)
                draw_platform(platform_list,screen,platform)
                
                #screen.blit(main.char,(main.x_position,main.y_position))
                
                #MOVEMENT
                main.movement += gravity
                main.collider.centery += main.movement
                main.y_position = main.collider.centery
                screen.blit(main.char,main.collider)
                
                
                #COLISION AND SCORE     
                for i in platform_list:
                    
                    if main.x_position > i.right:
                        main.score+=1 
                        platform_list.remove(i)
                        genes[bird_index].fitness += 10 #->when it passes gets +5
                        
                    
                    if main.collider.colliderect(i):
                        genes[bird_index].fitness += -10 #->when it hits the pipe, gets -1 fitness and removes
                        players.pop(bird_index)
                        networks.pop(bird_index)
                        #game_active = False
                        
                
                
                
                if main.collider.top <= -100 or main.collider.bottom >= 720:
                        genes[bird_index].fitness += -1 #->when it hits the pipe, gets -1 fitness and removes
                        players.pop(bird_index)
                        networks.pop(bird_index)
                        #game_active = False
                
                score(main.score,screen)
          
                
                pygame.display.update()
                clock.tick(120)
                

#AI

config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, "config.txt")

pop = neat.Population(config)
pop.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
pop.add_reporter(stats)

winner = pop.run(main,20)

print('\nBest genome:\n{!s}'.format(winner))


















